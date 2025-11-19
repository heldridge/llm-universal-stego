import argparse
import hashlib
import random

# import blessed
# Replace blessed with colorama
from colorama import init, Fore, Style
import os

from llama_cpp import Llama

import timeit
import utils
# term = blessed.Terminal()

# Initialize colorama
init(autoreset=True)


# Terminal class to replace blessed.Terminal()
class Terminal:
    @staticmethod
    def green(text):
        return Fore.GREEN + text + Style.RESET_ALL

    @staticmethod
    def red(text):
        return Fore.RED + text + Style.RESET_ALL

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

    Args:
        data (bytes): The bytes object to iterate over.

    Yields:
        int: Each bit in the bytes object, one by one (0 or 1).
    """
    for byte in data:
        for i in range(8):
            # Extract the bit at position i (from most significant to least)
            yield (byte >> (7 - i)) & 1


def first_bit_of_hash(h, b: bytes) -> int:
    return h(b).digest()[-1] & 1


def get_next_tokens(
        llm: Llama, prompt: str, target_bit: int, h, num_tokens=8, message="",
) -> tuple:
    seed = random.randint(0, 2 ** 64)
    increment = 0
    base_output = term.home + term.clear + message
    while True:
        if increment > 100:
            raise SystemError("Giving up after trying to generate next token with desired hash.")

        modified_base = base_output + f"attempt {increment + 1}\n"

        result = llm(prompt, max_tokens=num_tokens, seed=seed + increment,
                     temperature=1.3,
                     repeat_penalty=1.2
                     )

        output = result["choices"][0]["text"]

        # Tokenize the output to get exact token IDs
        output_tokens = llm.tokenize(output.encode('utf-8'))

        if len(output_tokens) >= num_tokens:
            # Use exactly num_tokens worth of tokens
            selected_tokens = output_tokens[:num_tokens]

            # Convert token IDs to bytes for hashing
            token_bytes = b''.join(str(tid).encode() for tid in selected_tokens)

            hash_works = first_bit_of_hash(h, token_bytes) == target_bit

            if hash_works:
                print(modified_base + prompt + term.green(output))
                return (output, increment + 1, selected_tokens)
            else:
                print(modified_base + prompt + term.red(output))

        increment += 1


def encode_bitstring(
        llm: Llama, initial_prompt: str, bitstring: bytes, h, tokens_per_bit=8
) -> str:
    """
    Encode a bitstring into LLM output using tokens.

    Args:
        llm: The language model
        initial_prompt: Starting prompt
        bitstring: Data to encode
        h: Hash function
        tokens_per_bit: Number of tokens to use per bit

    Returns:
        str: The encoded message
    """
    current_prompt = initial_prompt
    total_tries = 0

    start_time = timeit.default_timer()

    with term.fullscreen(), term.cbreak():
        base_output = (
                term.home + term.clear + f"bit 1/{len(bitstring) * 8}\t\t" + f"attempt {0}\n"
        )
        print(base_output + initial_prompt)
        for index, bit in enumerate(iter_bits(bitstring)):
            (next_prompt, tries, tokens) = get_next_tokens(
                llm,
                current_prompt,
                bit,
                h,
                tokens_per_bit,
                message=f"bit {index + 1}/{len(bitstring) * 8}\t\t",
            )
            current_prompt += next_prompt
            total_tries += tries

    end_time = timeit.default_timer()

    print(f"Elapsed: {round((end_time - start_time) * 1e6, 3)} µs")
    print(f"Elapsed: {round((end_time - start_time) * 1e3, 3)} ms")
    print(f"Elapsed per next token gen: {round((end_time - start_time) * 1e3 / total_tries, 3)} ms")

    print("total_tries: ", total_tries)
    print("len(bitstring): ", len(bitstring))
    print("avg try: ", total_tries / (len(bitstring) * 8))
    print("tokens_per_bit: ", tokens_per_bit)
    return current_prompt


def decode_bitstring(
        llm: Llama,
        encoded_message: str,
        initial_prompt: str,
        h,
        num_bits: int,
        tokens_per_bit: int
) -> tuple:
    """
    Decodes a bitstring from the encoded LLM output by re-tokenizing.

    Args:
        llm (Llama): The language model for tokenization.
        encoded_message (str): The full message output from the LLM after encoding.
        initial_prompt (str): The prompt that was used to generate the message.
        h: The hash function used to encode/decode bits.
        num_bits (int): The number of bits to decode.
        tokens_per_bit (int): Number of tokens used per bit during encoding.

    Returns:
        tuple: (decoded_bytes, decoded_string)
    """
    # Remove the initial prompt from the encoded message
    if not encoded_message.startswith(initial_prompt):
        raise ValueError("Encoded message does not start with the initial prompt.")

    # Get just the encoded portion
    encoded_text = encoded_message[len(initial_prompt):]

    # Re-tokenize the encoded text
    all_tokens = llm.tokenize(encoded_text.encode('utf-8'))

    bits = []

    for i in range(num_bits):
        # Extract tokens for this bit
        start_idx = i * tokens_per_bit
        end_idx = start_idx + tokens_per_bit

        if end_idx > len(all_tokens):
            raise ValueError(f"Not enough tokens in message for {num_bits} bits")

        tokens = all_tokens[start_idx:end_idx]

        # Convert token IDs to bytes for hashing (same as in encoding)
        token_bytes = b''.join(str(tid).encode() for tid in tokens)
        bit = first_bit_of_hash(h, token_bytes)
        bits.append(bit)

    print("bits: ", bits)
    # Convert list of bits to bytes
    decoded_bytes = bytearray()
    for b in range(0, len(bits), 8):
        byte = 0
        for i in range(8):
            if b + i < len(bits):
                byte = (byte << 1) | bits[b + i]
            else:
                byte = (byte << 1)
        decoded_bytes.append(byte)

    # Remove possible trailing null bytes (if the message was padded)
    return (decoded_bytes, decoded_bytes.rstrip(b"\x00").decode(errors="replace"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="The Alvarez hypothesis posits that the mass extinction of the dinosaurs and many other living things during the Cretaceous-Paleogene extinction event",
    )


    ###########################################

    args = parser.parse_args()

    h = hashlib.sha256

    # bitstring = b"hello"
    bitstring = b"abcdefg"
    bitstring = b"abc"
    print("bitstring type: ", type(bitstring))

    # replacing test bitstring with the elligator key exchange

    # set this to 2
    bitstring = utils.gen_elligator_bitstring(2)
    print("new bitstring type: ", type(bitstring))

    print("message to encode: " + str(bitstring))
    print("length: ", len(bitstring))

    tokens_per_bit_i = 8  # Changed from chars_per_bit_i
    print("tokens_per_bit_i: ", tokens_per_bit_i)


    # llm = Llama(model_path=args.model_path, verbose=False, n_ctx=0, device="cuda:0")
    # llm = Llama(model_path=args.model_path, n_gpu_layers=-1, verbose=False, n_threads=8, n_batch=512, use_mmap=True,  use_mlock=False,)
    # llm = Llama(model_path=args.model_path, n_gpu_layers=30, verbose=True, n_threads=8, n_ctx=0, use_mmap=True,  use_mlock=False,device="cuda:0")
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=-1,  # Offload all possible layers
        verbose=True,
        n_threads=8,  # Good for CPU fallback
        n_ctx=0,
        use_mmap=True,  # Good for large models
        use_mlock=False,  # Keep False to avoid memory issues
        # logits_all=True,
        device="cuda:0",
    )
    print("model loaded")

    max_retries = 10
    message = None
    for attempt in range(max_retries):
        try:
            message = encode_bitstring(llm, args.prompt, bitstring, h, tokens_per_bit_i)
            break  # Exit the loop if successful
        except SystemError as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Raising exception.")
                raise  # Re-raise the exception after max retries

    print("~~~~~~~\nMessage:\n~~~~~~~\n", message)

    # --- Decode the message back ---
    num_bits = len(bitstring) * 8
    (decoded_bytes, decoded) = decode_bitstring(
        llm,
        message,
        args.prompt,
        h,
        num_bits,
        tokens_per_bit_i
    )
    print("~~~~~~~\nDecoded message:\n~~~~~~~\n", decoded)

    assert decoded_bytes == bitstring