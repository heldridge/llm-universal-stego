import argparse
import hashlib
import random
import timeit
import os

# New imports for Hugging Face transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import infer_auto_device_map, init_empty_weights

# Replaced blessed with colorama
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# --- Terminal class to replace blessed.Terminal() ---
class Terminal:
    @staticmethod
    def green(text):
        return Fore.GREEN + text

    @staticmethod
    def red(text):
        return Fore.RED + text

    @property
    def clear(self):
        # Clears the screen and moves cursor to home
        return '\033[2J\033[H'

    @property
    def home(self):
        # Moves cursor to home position
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
    """Iterate over every bit in a bytes object."""
    for byte in data:
        for i in range(8):
            yield (byte >> (7 - i)) & 1


def first_bit_of_hash(h, b: bytes) -> int:
    """Calculates the hash and returns the last bit of the digest."""
    # Using the last bit as it's simple and effective for this purpose.
    return h(b).digest()[-1] & 1


def get_next_tokens(
        model, tokenizer, prompt: str, target_bit: int, h, num_chars=16, message="",
        batch_size=4,  # Add batch processing for parallel generation
) -> str:
    seed = random.randint(0, 2 ** 32 - 1)  # torch.manual_seed requires a 32-bit int
    increment = 0
    base_output = term.home + term.clear + message

    min_new_tokens = num_chars

    while True:
        if increment > 100:
            raise SystemError("Giving up after trying to generate next token with desired hash.")

        # Generate multiple candidates to check in sequence
        attempts_this_batch = min(batch_size, 100 - increment)
        
        for batch_idx in range(attempts_this_batch):
            current_increment = increment + batch_idx
            modified_base = base_output + f"attempt {current_increment + 1}\n"

            torch.manual_seed(seed + current_increment)

            # Tokenize the input prompt
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            # Generate text with optimizations
            outputs = model.generate(
                input_ids,
                max_new_tokens=min_new_tokens + 5,  # Generate a bit extra to ensure we have enough chars
                do_sample=True,
                temperature=1.3,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for faster generation
            )

            # Decode the newly generated part
            new_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

            if len(new_text) >= num_chars:
                final_str = new_text[:num_chars]

                hash_works = first_bit_of_hash(h, final_str.encode()) == target_bit

                # Display the attempt
                display_prompt = prompt if len(prompt) < 200 else "..." + prompt[-200:]
                if hash_works:
                    print(modified_base + display_prompt + term.green(final_str))
                    return (final_str, current_increment + 1)
                else:
                    print(modified_base + display_prompt + term.red(final_str))

        increment += attempts_this_batch


def encode_bitstring(
        model, tokenizer, initial_prompt: str, bitstring: bytes, h, chars_per_bit=16
) -> str:
    current_prompt = initial_prompt
    total_tries = 0
    total_bits = len(bitstring) * 8

    start_time = timeit.default_timer()

    with term.fullscreen(), term.cbreak():
        base_output = (
                term.home + term.clear + f"bit 1/{total_bits}\t\t" + f"attempt {0}\n"
        )
        print(base_output + initial_prompt)
        for index, bit in enumerate(iter_bits(bitstring)):
            (next_prompt, tries) = get_next_tokens(
                model,
                tokenizer,
                current_prompt,
                bit,
                h,
                chars_per_bit,
                message=f"bit {index + 1}/{total_bits}\t\t",
            )
            current_prompt += next_prompt
            total_tries += tries

    end_time = timeit.default_timer()

    print(f"\n--- Encoding Complete ---")
    print(f"Elapsed: {round((end_time - start_time), 3)} s")
    print(f"Elapsed per token generation: {round((end_time - start_time) * 1e3 / total_tries, 3)} ms")
    print(f"Total Tries: {total_tries}")
    print(f"Total Bits: {total_bits}")
    print(f"Average Tries per Bit: {round(total_tries / total_bits, 2)}")

    return current_prompt


def decode_bitstring(
        encoded_message: str, initial_prompt: str, h, num_bits: int, chars_per_bit=16
) -> str:
    if not encoded_message.startswith(initial_prompt):
        raise ValueError("Encoded message does not start with the initial prompt.")

    data = encoded_message[len(initial_prompt):]
    bits = []

    for i in range(num_bits // 8):
        for j in range(8):
            chunk_start = (i * 8 + j) * chars_per_bit
            chunk_end = chunk_start + chars_per_bit
            chunk = data[chunk_start:chunk_end]
            bit = first_bit_of_hash(h, chunk.encode())
            bits.append(bit)

    # Convert list of bits to bytes
    decoded_bytes = bytearray()
    for b in range(0, len(bits), 8):
        byte = 0
        for i in range(8):
            byte = (byte << 1) | bits[b + i]
        decoded_bytes.append(byte)

    return (decoded_bytes, decoded_bytes.rstrip(b"\x00").decode(errors="replace"))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    parser = argparse.ArgumentParser(description="Encode a secret message into LLM-generated text using transformers.")
    parser.add_argument("model_id", type=str,
                        help="Hugging Face model ID (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct').")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="The Alvarez hypothesis posits that the mass extinction of the dinosaurs and many other living things during the Cretaceous-Paleogene extinction event",
    )
    args = parser.parse_args()


    # --- Message to encode ---
    h = hashlib.sha256

    # bitstring = b"hello"
    bitstring = b"abcdefg"
    bitstring = b"abc"

    # replacing test bitstring with the elligator key exchange
    # bitstring = utils.gen_elligator_bitstring()
    print("new bitstring type: ", type(bitstring))

    print("message to encode: " + str(bitstring))
    print("length: ", len(bitstring))

    chars_per_bit_i = 32
    print("chars_per_bit_i: ", chars_per_bit_i)

    print(f"Message to encode: '{bitstring.decode()}' ({len(bitstring)} bytes)")
    print(f"Using model: {args.model_id}")
    print("-------------------------------------------")

    print("Loading model...")

    # Configure 4-bit quantization optimized
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # float16 is optimal
        bnb_4bit_use_double_quant=True,  # Additional compression for better memory efficiency
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Try to load with Flash Attention 2 if available, fallback to standard attention
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",  # Significantly faster attention mechanism
        )
        print("Model loaded with Flash Attention 2 - optimal performance enabled.")
    except Exception as e:
        print(f"Flash Attention 2 not available ({e}), loading with standard attention...")
        print("To enable Flash Attention 2, install: pip install flash-attn --no-build-isolation")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            quantization_config=quantization_config,
        )
    
    print("Model loaded successfully.")
    
    # Display GPU memory usage
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # --- Main Encoding/Decoding Logic ---
    # TODO: record how many next_token gens were done if failed
    max_retries = 10
    message = ""
    for attempt in range(max_retries):
        try:
            message = encode_bitstring(model, tokenizer, args.prompt, bitstring, h, chars_per_bit_i)
            break
        except SystemError as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Raising exception.")
                raise

    print("\n~~~~~~~\nFinal Generated Message:\n~~~~~~~\n", message)

    # --- Decode the message back ---
    num_bits = len(bitstring) * 8
    (decoded_bytes, decoded_str) = decode_bitstring(
        message,
        args.prompt,
        h,
        num_bits,
        chars_per_bit_i
    )
    print("\n~~~~~~~\nDecoded message:\n~~~~~~~\n", decoded_str)

    # Final verification
    assert decoded_bytes == bitstring