import argparse
import hashlib
import random
import math
import os
import copy
import blake3
import numpy as np
import concurrent.futures

# Set tokenizers parallelism before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Replace blessed with colorama
from colorama import init, Fore, Style

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, BitsAndBytesConfig

import timeit
import utils

tokens_per_bit_i = 4
max_attempts_param = 250

write_to_file = True
LOG_OUTPUT_FILE = "bench.txt"
MSG_OUTPUT_FILE = "msg.txt"
CHUNCK_OUTPUT_FILE = "msg_chunks.txt"

def replace_slashes(s: str) -> str:
    return s.replace("/", "_")


# Initialize colorama
init(autoreset=True)


# Terminal class to replace blessed. Terminal()
class Terminal:
    @staticmethod
    def green(text):
        return Fore.GREEN + text + Style.RESET_ALL

    @staticmethod
    def red(text):
        return Fore.RED + text + Style.RESET_ALL

    @staticmethod
    def yellow(text):
        return Fore.YELLOW + text + Style.RESET_ALL

    @staticmethod
    def cyan(text):
        return Fore.CYAN + text + Style.RESET_ALL

    @property
    def clear(self):
        return '\033[2J\033[H'

    @property
    def home(self):
        return '\033[H'

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fullscreen(self):
        return self._DummyContext()

    def cbreak(self):
        return self._DummyContext()


term = Terminal()


def iter_bits(data: bytes):
    """
    Iterate over every bit in a bytes object.
    """
    for byte in data:
        for i in range(8):
            yield (byte >> (7 - i)) & 1


def first_bit_of_hash(h, b: bytes) -> int:
    return h(b).digest()[-1] & 1


def copy_cache(cache):
    """Helper function to safely copy a cache object."""
    if cache is None:
        return None

    if isinstance(cache, DynamicCache):
        # Create a new DynamicCache and copy the key-value pairs
        new_cache = DynamicCache()
        legacy = cache.to_legacy_cache()
        if legacy is not None:
            new_cache = DynamicCache.from_legacy_cache(legacy)
        return new_cache
    else:
        # If it's already a tuple (legacy format), convert it to DynamicCache
        return DynamicCache.from_legacy_cache(cache)


def apply_sampling_params(logits, temperature, top_p):
    """
    Applies temperature and top_p sampling to logits.
    """
    if temperature != 1.0:
        logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )
        # Find cutoff
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create mask for indices to remove
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[..., indices_to_remove] = float("-inf")

    return logits


def check_tokenization_stability(tokenizer, input_ids):
    """
    Verifies that decoding and re-encoding the input_ids results in the exact same sequence.
    This ensures that tokenizer merges do not alter the sequence perceived by the decoder.
    """
    text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    re_encoded = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')

    if re_encoded.device != input_ids.device:
        re_encoded = re_encoded.to(input_ids.device)

    if re_encoded.shape != input_ids.shape:
        return False

    return torch.equal(input_ids, re_encoded)


def embed_bit_with_rejection_sampling(
        model,
        tokenizer,
        current_input_ids: torch.Tensor,
        past_key_values,
        attention_mask: torch.Tensor,
        target_bit: int,
        h,
        num_tokens: int,
        entropy_threshold: float,
        temperature: float,
        top_p: float,
        vocab_size: int,
        max_attempts: int = 500
) -> tuple:
    """
    Attempt to embed a bit using Rejection Sampling.

    Constraints for acceptance:
    1.  No EOS tokens (sequence must continue)
    2.  Tokenization Stability (re-encoded tokens match generated tokens)
    3.  Entropy of chunk >= entropy_threshold
    4.  Hash(tokens) == target_bit
    """

    initial_past = copy_cache(past_key_values)
    initial_attn = attention_mask.clone() if attention_mask is not None else None

    # Pre-fetch EOS id to avoid lookups inside the loop
    eos_token_id = tokenizer.eos_token_id

    attempts = 0

    while attempts < max_attempts:
        attempts += 1

        # Reset state for this attempt
        temp_input_ids = current_input_ids
        temp_past = copy_cache(initial_past)
        temp_attn = initial_attn.clone() if initial_attn is not None else None

        sampled_tokens = []
        chunk_entropy = 0.0
        contains_eos = False  # Flag to track EOS detection

        # Sample num_tokens
        for token_idx in range(num_tokens):
            with torch.no_grad():
                if temp_past is not None and isinstance(temp_past, DynamicCache) and temp_past.get_seq_length() > 0:
                    output = model(
                        temp_input_ids[:, -1:],
                        past_key_values=temp_past,
                        attention_mask=temp_attn,
                        use_cache=True
                    )
                else:
                    output = model(temp_input_ids, use_cache=True)

                logits = output.logits[:, -1, : vocab_size]
                logits = apply_sampling_params(logits, temperature, top_p)
                probs = torch.softmax(logits, dim=-1)

                token = torch.multinomial(probs, num_samples=1)

                # CHECK 1: Immediate EOS Check
                if token.item() == eos_token_id:
                    contains_eos = True
                    break  # Break inner loop, will be caught by check below

                # Calculate entropy for this token
                token_prob = probs[0, token.item()].item()
                token_ent = -math.log2(token_prob) if token_prob > 0 else 0
                chunk_entropy += token_ent

                sampled_tokens.append(token.item())

                temp_input_ids = torch.cat([temp_input_ids, token], dim=-1)
                temp_past = output.past_key_values
                temp_attn = torch.cat([temp_attn, temp_attn.new_ones((temp_attn.shape[0], 1))], dim=-1)

        # If we broke out early due to EOS, retry immediately
        if contains_eos:
            continue

        # CHECK 2: Tokenization Stability
        if not check_tokenization_stability(tokenizer, temp_input_ids):
            continue

        # CHECK 3: Entropy Threshold
        if chunk_entropy < entropy_threshold:
            # Chunk is too predictable, reject it
            continue

        # CHECK 4: Hash Constraint
        token_bytes = b''.join(str(tid).encode() for tid in sampled_tokens)
        hash_works = first_bit_of_hash(h, token_bytes) == target_bit

        if hash_works:
            output_text = tokenizer.decode(sampled_tokens, skip_special_tokens=False)
            return (
                output_text,
                sampled_tokens,
                temp_input_ids,
                temp_past,
                temp_attn,
                True,
                chunk_entropy,
                attempts * num_tokens
            )

    # Failed after max_attempts
    output_text = tokenizer.decode(sampled_tokens, skip_special_tokens=False)
    return (
        output_text,
        sampled_tokens,
        temp_input_ids,
        temp_past,
        temp_attn,
        False,
        0.0,
        attempts * num_tokens
    )


def encode_bitstring(
        model,
        tokenizer,
        initial_prompt: str,
        bitstring: bytes,
        h,
        tokens_per_bit=8,
        entropy_threshold=10.0,
        temperature=1.0,
        top_p=1.0,
        target_output_tokens=None,
        quiet=False
) -> tuple:
    """
    Encode a bitstring into text generation.

    Args:
        target_output_tokens: Total number of tokens desired in output (excluding prompt).
                            If None or less than required, defaults to minimum needed.
                            If greater, additional unbiased tokens are generated after embedding.
    """
    # Tokenize initial prompt
    input_ids = tokenizer.encode(
        initial_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
        truncation=False
    ).to(model.device)

    # Ensure prompt itself is stable before starting
    if not check_tokenization_stability(tokenizer, input_ids):
        if not quiet:
            print(term.yellow("Warning: Initial prompt is not stable under re-tokenization.  Stabilizing..."))
        txt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        input_ids = tokenizer.encode(txt, add_special_tokens=False, return_tensors='pt').to(model.device)

    attention_mask = torch.ones_like(input_ids)
    past_key_values = None

    vocab_size = len(tokenizer)

    bits_to_embed = list(iter_bits(bitstring))
    total_bits = len(bits_to_embed)
    current_bit_index = 0

    # Calculate minimum tokens needed for embedding
    min_tokens_needed = total_bits * tokens_per_bit

    # Determine actual target
    if target_output_tokens is None or target_output_tokens < min_tokens_needed:
        actual_target_tokens = min_tokens_needed
        extra_tokens_needed = 0
    else:
        actual_target_tokens = target_output_tokens
        extra_tokens_needed = target_output_tokens - min_tokens_needed

    start_time = timeit.default_timer()

    if not quiet:
        with term.fullscreen(), term.cbreak():
            base_output = (
                    term.home + term.clear + f"Embedded:  0/{total_bits} bits\n"
                    + f"Target tokens: {actual_target_tokens} (embedding: {min_tokens_needed}, extra: {extra_tokens_needed})\n"
            )
            print(base_output + initial_prompt)
    else:
        print(
            f"Encoding {total_bits} bits into {actual_target_tokens} tokens...  (use without --quiet to see progress)")

    generation_count = 0
    total_tokens_generated = 0
    total_tokens_tried=0

    # Phase 1: Embed all bits
    while current_bit_index < total_bits:
        generation_count += 1
        target_bit = bits_to_embed[current_bit_index]

        if not quiet:
            status_msg = f"Embedded: {current_bit_index}/{total_bits} bits | Tokens: {total_tokens_generated}/{actual_target_tokens} | Gen #{generation_count} | Target: {target_bit}\n"
            print(term.home + term.clear + status_msg + f"[ATTEMPTING bit {current_bit_index + 1}/{total_bits}]")

        (
            output_text,
            tokens,
            input_ids,
            past_key_values,
            attention_mask,
            success,
            final_entropy,
            tokens_tried
        ) = embed_bit_with_rejection_sampling(
            model,
            tokenizer,
            input_ids,
            past_key_values,
            attention_mask,
            target_bit,
            h,
            tokens_per_bit,
            entropy_threshold,
            temperature,
            top_p,
            vocab_size,
            max_attempts=max_attempts_param
        )

        total_tokens_tried += tokens_tried

        if success:
            total_tokens_generated += tokens_per_bit
            if not quiet:
                print(term.green(
                    f"✓ Embedded bit {current_bit_index + 1} (value={target_bit}, entropy={final_entropy:.2f})"))
                print(term.green(output_text))

            current_bit_index += 1

            if quiet and current_bit_index % 8 == 0:
                print(
                    f"  Progress: {current_bit_index}/{total_bits} bits embedded ({100 * current_bit_index // total_bits}%)")
        else:
            error_msg = (f"Could not embed bit {current_bit_index + 1} after max attempts")
            if write_to_file:
                with open(LOG_OUTPUT_FILE, "a+") as f:
                    f.write(f"Could not embed bit {current_bit_index + 1} after max attempts" + "\n")

                    end_time = timeit.default_timer()
                    print(f"Failed Elapsed: {round((end_time - start_time), 3)} secs")
                    print(f"Total tokens tried: {total_tokens_tried}")
                    f.write(f"Failed Elapsed: {round((end_time - start_time), 3)} secs\n")
                    f.write(f"Total tokens tried: {total_tokens_tried}\n")
            raise SystemError(error_msg)

    # Phase 2: Generate extra tokens without embedding constraints (if needed)
    if extra_tokens_needed > 0:
        if not quiet:
            print(term.cyan(f"\nGenerating {extra_tokens_needed} additional unbiased tokens..."))

        for i in range(extra_tokens_needed):
            with torch.no_grad():
                if past_key_values is not None and isinstance(past_key_values,
                                                              DynamicCache) and past_key_values.get_seq_length() > 0:
                    output = model(
                        input_ids[:, -1:],
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
                else:
                    output = model(input_ids, use_cache=True)

                logits = output.logits[:, -1, :vocab_size]
                logits = apply_sampling_params(logits, temperature, top_p)
                probs = torch.softmax(logits, dim=-1)

                token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, token], dim=-1)
                past_key_values = output.past_key_values
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                                           dim=-1)
                total_tokens_generated += 1

                if not quiet and (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{extra_tokens_needed} extra tokens")
        total_tokens_tried += extra_tokens_needed

    # --- NEW: COMPLETION PHASE ---
    if not quiet:
        print(term.cyan("\nAll bits embedded. Finishing sentence generation..."))
    
    finished_sentence = False
    max_completion_tokens = 256  # Safety limit to prevent infinite loops
    completion_count = 0

    while not finished_sentence and completion_count < max_completion_tokens:
        with torch.no_grad():
            if past_key_values is not None and isinstance(past_key_values, DynamicCache) and past_key_values.get_seq_length() > 0:
                output = model(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True
                )
            else:
                output = model(input_ids, use_cache=True)

            logits = output.logits[:, -1, :vocab_size]
            logits = apply_sampling_params(logits, temperature, top_p)
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            total_tokens_generated += 1
            
            # Append token to inputs
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = output.past_key_values
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            
            completion_count += 1
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                finished_sentence = True
                if not quiet:
                    print(term.cyan("EOS reached."))
    # -----------------------------
    
    full_message = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
    end_time = timeit.default_timer()

    print(f"\n{'=' * 60}")
    print(f"Embed Elapsed: {round((end_time - start_time), 3)} secs")
    print(f"Total generations: {generation_count}")
    print(f"Total tokens generated:  {total_tokens_generated}")
    print(f"Bitstring length: {len(bitstring)} bytes ({total_bits} bits)")
    print(f"Total tokens tried: {total_tokens_tried}")
    print(f"All {total_bits} bits successfully embedded!")
    print(f"{'=' * 60}\n")

    if write_to_file:
        with open(LOG_OUTPUT_FILE, "a+") as f:
            f.write(f"\n{'=' * 60}" + "\n")
            f.write(f"Embed Elapsed: {round((end_time - start_time), 3)} secs" + "\n")
            f.write(f"Total generations: {generation_count}" + "\n")
            f.write(f"Total tokens generated: {total_tokens_generated}" + "\n")
            f.write(f"Bitstring length: {len(bitstring)} bytes ({total_bits} bits)" + "\n")
            f.write(f"All {total_bits} bits successfully embedded!" + "\n")
            f.write(f"Total tokens tried: {total_tokens_tried}\n")
            f.write(f"\n{'-' * 60}" + "\n")

    return (full_message, None)

def decode_bitstring(
        tokenizer,
        initial_prompt: str,
        h,
        full_message: str,
        tokens_per_bit: int,
        num_bits_to_decode: int = None,
        quiet=False,
        write_to_file=False  # Added missing arg based on context
) -> tuple:
    start_time = timeit.default_timer()

    # 1. Tokenize the Prompt (Context)
    prompt_ids = tokenizer.encode(
        initial_prompt,
        return_tensors="pt",
        add_special_tokens=False
    )

    # 2. Tokenize the Full Message
    full_ids = tokenizer.encode(
        full_message,
        return_tensors="pt",
        add_special_tokens=False
    )

    if full_ids.shape[1] < prompt_ids.shape[1]:
        raise ValueError("Full message is shorter than prompt")

    # Extract generated portion
    generated_tensor = full_ids[0, prompt_ids.shape[1]:]

    # Pre-calculate the maximum chunks we can possibly decode
    max_chunks = len(generated_tensor) // tokens_per_bit
    num_chunks = min(num_bits_to_decode, max_chunks) if num_bits_to_decode else max_chunks

    # Prepare the input matrix (Batching)
    # Convert to Numpy CPU immediately to avoid PyTorch .item() overhead
    # Shape: [num_chunks, tokens_per_bit]
    relevant_ids = generated_tensor[:num_chunks * tokens_per_bit].cpu().numpy()
    chunk_matrix = relevant_ids.reshape(num_chunks, tokens_per_bit)


    unique_tokens = np.unique(chunk_matrix)
    vocab_cache = {t: str(t).encode() for t in unique_tokens}

    # Define the worker function for parallel execution
    def process_chunk(chunk_row):
        # 1. Join bytes using the fast lookup table (No str().encode() calls)
        token_bytes = b''.join(vocab_cache[t] for t in chunk_row)
        # 2. Hash and extract bit
        return first_bit_of_hash(h, token_bytes)

    decoded_bits = []
    if num_chunks > 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # map preserves order
            decoded_bits = list(executor.map(process_chunk, chunk_matrix))
    else:
        decoded_bits = []


    if not quiet:
        print(f"\nDecoding {num_chunks} chunks...")
        for i, bit in enumerate(decoded_bits):
            # Reconstruct text for display only
            chunk_ids = chunk_matrix[i]
            chunk_txt = tokenizer.decode(chunk_ids)
            print(f"Chunk {i}: '{chunk_txt}' -> Bit: {bit}")

    if write_to_file:
        with open(CHUNCK_OUTPUT_FILE, "a+") as f:
            f.write(f"\n\n\n\n +++++++++++ NEW MSG ++++++++++++++++++++++++++\n\n{full_message}\n\n")
            for i, bit in enumerate(decoded_bits):
                chunk_txt = tokenizer.decode(chunk_matrix[i])
                f.write(f"Chunk {i}: '{chunk_txt}' -> Bit:  {bit}\n")
            f.write(f"Decoded bits: {decoded_bits}\n")

    # Reconstruct Bytes from Bits
    decoded_bytes = bytearray()
    for b in range(0, len(decoded_bits), 8):
        byte = 0
        for i in range(8):
            if b + i < len(decoded_bits):
                byte = (byte << 1) | decoded_bits[b + i]
            else:
                byte = (byte << 1)
        decoded_bytes.append(byte)

    decoded_string = decoded_bytes.rstrip(b"\x00").decode(errors="replace")

    end_time = timeit.default_timer()
    print(f"Recover Elapsed: {round((end_time - start_time), 15)} secs")

    return (decoded_bytes, decoded_string, decoded_bits)

def decode_bitstring_to_bits(
        tokenizer,
        initial_prompt: str,
        h,
        full_message: str,
        tokens_per_bit: int,
        num_bits_to_decode: int = None,
) -> tuple:
    """
    Decodes a bitstring ONLY using the tokenizer.
    It assumes every chunks of `tokens_per_bit` in the generated text
    is a valid bit carrier.

    Args:
        num_bits_to_decode: Number of bits to decode. If None, decodes all available chunks.
                           Extra tokens beyond what's needed are ignored.
    """
    start_time = timeit.default_timer()


    # 1. Tokenize the Prompt (Context)
    prompt_ids = tokenizer.encode(
        initial_prompt,
        return_tensors="pt",
        add_special_tokens=False
    )

    # 2. Tokenize the Full Message
    full_ids = tokenizer.encode(
        full_message,
        return_tensors="pt",
        add_special_tokens=False
    )

    # 3. Find start of generation
    if full_ids.shape[1] < prompt_ids.shape[1]:
        raise ValueError("Full message is shorter than prompt")

    # We don't perform deep equality checks on device since we are just doing logic
    # But we check if the prompt prefix matches
    if not torch.equal(full_ids[0, :prompt_ids.shape[1]].cpu(), prompt_ids[0].cpu()):
        print(term.yellow(
            "Warning: Prompt tokens do not match exactly. Decoding might fail due to tokenizer merge at boundary. "))

    # Extract the generated portion
    generated_ids = full_ids[0, prompt_ids.shape[1]:]

    bits = []

    # Iterate through generated tokens in chunks
    # Note: We discard trailing tokens that don't fit into a full chunk
    max_chunks = len(generated_ids) // tokens_per_bit

    # Determine how many chunks to actually decode
    if num_bits_to_decode is not None:
        num_chunks = min(num_bits_to_decode, max_chunks)
    else:
        num_chunks = max_chunks


    for i in range(num_chunks):
        chunk_start = i * tokens_per_bit
        chunk_end = chunk_start + tokens_per_bit
        chunk_ids = generated_ids[chunk_start:chunk_end]

        # Calculate Hash
        token_bytes = b''.join(str(tid.item()).encode() for tid in chunk_ids)
        decoded_bit = first_bit_of_hash(h, token_bytes)
        bits.append(decoded_bit)
    end_time = timeit.default_timer()
    print(f" Recover Elapsed: {round((end_time - start_time), 15)} secs")
    return (bits)

def calculate_perplexity(model, tokenizer, text: str, context: str = None, min_tokens: int = 2) -> float:
    """
    Calculate perplexity of generated text given optional context.

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: The text to measure perplexity for
        context:   Optional context/prompt (if provided, only measures perplexity of text after context)
        min_tokens:   Minimum number of tokens required (default: 2)

    Returns:
        perplexity:   The perplexity score (or None if text is too short)
    """
    model.eval()

    # Tokenize the full sequence
    if context:
        full_text = context + text
        context_ids = tokenizer.encode(context, return_tensors="pt", add_special_tokens=False).to(model.device)
        context_len = context_ids.shape[1]
    else:
        full_text = text
        context_len = 0

    input_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    # Check if we have enough tokens
    eval_length = input_ids.shape[1] - context_len
    if eval_length < min_tokens:
        print(f"Warning: Text too short ({eval_length} tokens), need at least {min_tokens}")
        return None

    # We need at least context_len + 1 tokens total
    if input_ids.shape[1] < context_len + 1:
        return None

    try:
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Only measure perplexity on the generated part (after context)
        if context_len > 0:
            # Make sure we have tokens to evaluate
            if shift_logits.shape[1] <= context_len:
                return None
            shift_logits = shift_logits[:, context_len:, :]
            shift_labels = shift_labels[:, context_len:]

        # Check if we still have tokens after slicing
        if shift_labels.shape[1] == 0:
            return None

        # Calculate cross-entropy loss for each token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss value: {loss.item()}")
            return None

        avg_loss = loss.item()

        # Perplexity is exp(average negative log likelihood)
        perplexity = math.exp(avg_loss)

        # Check for overflow
        if math.isnan(perplexity) or math.isinf(perplexity):
            print(f"Warning: Invalid perplexity (loss={avg_loss})")
            return None

        return perplexity

    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None


def calculate_chunk_perplexities(model, tokenizer, full_message: str, prompt: str, tokens_per_bit: int):
    """Calculate perplexity for each embedded chunk WITH proper context."""
    model.eval()

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    full_ids = tokenizer.encode(full_message, return_tensors="pt", add_special_tokens=False).to(model.device)
    generated_ids = full_ids[0, prompt_ids.shape[1]:]

    num_chunks = len(generated_ids) // tokens_per_bit
    perplexities = []

    print(f"\nCalculating perplexity for {num_chunks} chunks...")
    print(f"Tokens per chunk: {tokens_per_bit}")

    for i in range(num_chunks):
        chunk_start = i * tokens_per_bit
        chunk_end = chunk_start + tokens_per_bit
        chunk_ids = generated_ids[chunk_start:chunk_end]

        # IMPORTANT: Include ALL context up to this chunk
        # This includes the prompt AND all previously generated text
        context_end_pos = prompt_ids.shape[1] + chunk_start
        context_ids = full_ids[0, :  context_end_pos]

        # The chunk we want to evaluate
        eval_ids = full_ids[0, context_end_pos:  context_end_pos + tokens_per_bit]

        # We need context + at least 1 token to evaluate
        if eval_ids.shape[0] < 1:
            print(f"Chunk {i}:  Skipped (no tokens to evaluate)")
            perplexities.append(None)
            continue

        # Concatenate context + chunk for evaluation
        full_sequence = torch.cat([context_ids, eval_ids]).unsqueeze(0)

        try:
            with torch.no_grad():
                outputs = model(full_sequence)
                logits = outputs.logits

            # We only calculate loss on the CHUNK tokens, not the context
            # logits shape: [1, seq_len, vocab_size]
            # We want to predict tokens at positions [context_len :   context_len + chunk_len]

            context_len = context_ids.shape[0]

            # Get logits for predicting the chunk tokens
            # logits[0, i] predicts token at position i+1
            # So logits[0, context_len-1 :   context_len+chunk_len-1] predicts the chunk
            chunk_pred_logits = logits[0, context_len - 1:context_len + tokens_per_bit - 1, :]
            chunk_labels = eval_ids

            # Make sure dimensions match
            if chunk_pred_logits.shape[0] != chunk_labels.shape[0]:
                print(f"Chunk {i}: Dimension mismatch, skipping")
                perplexities.append(None)
                continue

            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(chunk_pred_logits, chunk_labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Chunk {i}: Invalid loss")
                perplexities.append(None)
                continue

            ppl = math.exp(loss.item())

            if math.isnan(ppl) or math.isinf(ppl):
                print(f"Chunk {i}: Invalid perplexity (loss={loss.item():. 4f})")
                perplexities.append(None)
            else:
                chunk_text = tokenizer.decode(chunk_ids)
                print(f"Chunk {i}:   PPL = {ppl:.2f} | Text:  '{chunk_text}'")
                perplexities.append(ppl)

        except Exception as e:
            print(f"Chunk {i}: Error - {e}")
            perplexities.append(None)

    # Filter out None values for statistics
    valid_ppls = [p for p in perplexities if p is not None]
    if valid_ppls:
        print(f"\nChunk Perplexity Statistics:")
        print(f"  Mean: {sum(valid_ppls) / len(valid_ppls):.2f}")
        print(f"  Min: {min(valid_ppls):.2f}")
        print(f"  Max: {max(valid_ppls):.2f}")

    return perplexities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path or HuggingFace model ID")
    parser.add_argument("--prompt", "-p", type=str,
                        default="The Alvarez hypothesis posits that the mass extinction of the dinosaurs and many other living things during the Cretaceous-Paleogene extinction event")
    parser.add_argument("--entropy-threshold", type=float, default=10.0,
                        help="Minimum entropy threshold before embedding bits")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p parameter")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress debug output")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--max-memory",
        type=str,
        default=None,
        help="Maximum memory per GPU (e.g., '9GiB' for 3080 to leave room for operations)"
    )
    parser.add_argument("--tokens-per-bit-arg", type=int, default=8, help="Tokens per bit argument")
    parser.add_argument("--target-output-tokens", type=int, default=None,
                        help="Target total number of output tokens (excluding prompt). If greater than needed for embedding, extra tokens are generated without bias.")

    args = parser.parse_args()
    #h = hashlib.sha256
    h = blake3.blake3

    bitstring = utils.gen_elligator_bitstring(3)
    # bitstring = b"abc"
    #bitstring = b"a"
    #bitstring = bitstring[:9]
    #bitstring = bitstring[:1]
    #bitstring = b"abcdefg"

    if args.tokens_per_bit_arg:
        print("args.tokens_per_bit_arg:  ", args.tokens_per_bit_arg)
        tokens_per_bit_i = args.tokens_per_bit_arg

    print("message to encode: " + str(bitstring))
    print("length:  ", len(bitstring))
    print("tokens_per_bit_i: ", tokens_per_bit_i)

    print(f"Loading model:  {args.model_path}")

    bit_flag = ""
    if args.load_in_4bit:
        bit_flag = "_4bit"
    if args.load_in_8bit:
        bit_flag = "_8bit"

    entropy_flag = "_entropy" + str(args.entropy_threshold)
    max_attempts_param_str = "_maxattempts" + str(max_attempts_param)

    tokens_output_str = ""

    if args.target_output_tokens:
        tokens_output_str = "_to" + str(args.target_output_tokens)

    LOG_OUTPUT_FILE = replace_slashes(args.model_path) + entropy_flag + "_token" + str(
        tokens_per_bit_i) + bit_flag + max_attempts_param_str + tokens_output_str + "_" + LOG_OUTPUT_FILE
    MSG_OUTPUT_FILE = replace_slashes(args.model_path) + entropy_flag + "_token" + str(
        tokens_per_bit_i) + bit_flag + max_attempts_param_str + tokens_output_str +"_" + MSG_OUTPUT_FILE
    CHUNCK_OUTPUT_FILE = replace_slashes(args.model_path) + entropy_flag + "_token" + str(
        tokens_per_bit_i) + bit_flag + max_attempts_param_str + tokens_output_str +"_" + CHUNCK_OUTPUT_FILE
    if write_to_file:
        with open(LOG_OUTPUT_FILE, "a+") as f:
            f.write(f"Loading model: {args.model_path}" + "\n")
            f.write("message to encode: " + str(bitstring) + "\n")
            f.write("length: " + str(len(bitstring)) + "\n")
            f.write("tokens_per_bit_i: " + str(tokens_per_bit_i) + "\n")
            if args.target_output_tokens:
                f.write("target_output_tokens: " + str(args.target_output_tokens) + "\n")

    if args.cpu:
        print("Running on CPU")
        load_kwargs = {"device_map": "cpu", "dtype": torch.float32}
    else:
        load_kwargs = {"device_map": "auto", "dtype": torch.float16}
        if args.load_in_4bit or args.load_in_8bit:
            if args.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif args.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = quantization_config

        if args.max_memory:
            print("args.max_memory: ", args.max_memory)
            max_memory_mapping = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    max_memory_mapping[i] = args.max_memory
            load_kwargs["max_memory"] = {0: args.max_memory}
            max_memory_mapping["cpu"] = "64GiB"  # Allow CPU offload
            load_kwargs["max_memory"] = max_memory_mapping

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU:  {torch.cuda.get_device_name(0)}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.compile and not (args.load_in_4bit or args.load_in_8bit):
        print("Compiling model...")
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")

    print(f"Model loaded successfully!")

    max_retries = 20
    message = None

    for attempt in range(max_retries):
        try:
            (message, _) = encode_bitstring(
                model,
                tokenizer,
                args.prompt,
                bitstring,
                h,
                tokens_per_bit_i,
                entropy_threshold=args.entropy_threshold,
                temperature=args.temperature,
                top_p=args.top_p,
                target_output_tokens=args.target_output_tokens,
                quiet=args.quiet
            )
            break
        except SystemError as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise

    print("~~~~~~~\nFull Message:\n~~~~~~~")
    print(message)
    print()

    if write_to_file:
        with open(MSG_OUTPUT_FILE, "a+") as f:
            f.write("-----------------------\n\n" + message + "\n\n\n")

    # --- Decode the message back (MODEL FREE) ---
    # Calculate how many bits we need to decode
    num_bits = len(bitstring) * 8

    (decoded_bytes, decoded_string, valid_bits) = decode_bitstring(
        tokenizer,
        args.prompt,
        h,
        message,
        tokens_per_bit_i,
        num_bits_to_decode=num_bits,
        quiet=args.quiet
    )
    print("\n~~~~~~~\nDecoded message:\n~~~~~~~")
    print(decoded_string)
    print("\n~~~~~~~\nDecoded bytes:\n~~~~~~~")
    print(decoded_bytes)
    print()
    bits_to_embed = list(iter_bits(bitstring))
    decoded_bits_to_embed = list(iter_bits(decoded_bytes))

    if len(valid_bits) == num_bits:
        if decoded_bytes == bitstring:
            print(term.green("✓ Successfully decoded!  Bytes match original bitstring."))
        else:
            print(term.red("✗ Decoding mismatch! Bytes don't match original bitstring."))
            print(f"Expected: {bitstring}")
            print(f"Got:      {decoded_bytes}")
            print("expected bits:  ", bits_to_embed)
            print("decoded bits: ", decoded_bits_to_embed)
            if write_to_file:
                with open(LOG_OUTPUT_FILE, "a+") as f:
                    f.write(f"\n{'#' * 60}" + "\n")
                    f.write("✗ Decoding mismatch!  Bytes don't match original bitstring." + "\n")
                    f.write(f"Expected: {bitstring}" + "\n")
                    f.write(f"Got:      {decoded_bytes}" + "\n")
                    f.write("expected bits: " + str(bits_to_embed) + "\n")
                    f.write("decoded bits: " + str(decoded_bits_to_embed) + "\n")
    else:
        print(term.red(f"✗ Bit count mismatch:  Expected {num_bits} bits, got {len(valid_bits)} bits"))
        print("expected bits: ", bits_to_embed)
        print("decoded bits: ", decoded_bits_to_embed)
        if write_to_file:
            with open(LOG_OUTPUT_FILE, "a+") as f:
                f.write(f"\n{'#' * 60}" + "\n")
                f.write(f"✗ Bit count mismatch: Expected {num_bits} bits, got {len(valid_bits)} bits" + "\n")
                f.write(f"Expected: {bitstring}" + "\n")
                f.write(f"Got:      {decoded_bytes}" + "\n")
                f.write("expected bits: " + str(bits_to_embed) + "\n")
                f.write("decoded bits: " + str(decoded_bits_to_embed) + "\n")
