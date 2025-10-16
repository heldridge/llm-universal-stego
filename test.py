import argparse
import hashlib
import random

import blessed
from llama_cpp import Llama

import timeit

term = blessed.Terminal()


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
    llm: Llama, prompt: str, target_bit: int, h, num_chars=16, message="",
) -> str:
    seed = random.randint(0, 2**64)
    increment = 0
    base_output = term.home + term.clear + message
    while True:
        if increment > 50:
            raise SystemError("Giving up after trying to generate next token with desired hash.")            

        modified_base = base_output + f"attempt {increment + 1}\n"

        result = llm(prompt, max_tokens=num_chars, seed=seed + increment)
        output = result["choices"][0]["text"]
        if len(output) >= num_chars:
            final_str = output[:num_chars]

            hash_works = first_bit_of_hash(h, final_str.encode()) == target_bit

            if hash_works:
                print(modified_base + prompt + term.green(final_str))
                return (final_str, increment+1)
            else:
                print(modified_base + prompt + term.red(final_str))

        increment += 1


def encode_bitstring(
    llm: Llama, initial_prompt: str, bitstring: bytes, h, chars_per_bit=16
) -> str:
    current_prompt = initial_prompt
    total_tries = 0

    start_time = timeit.default_timer()


    with term.fullscreen(), term.cbreak():
        base_output = (
            term.home + term.clear + f"bit 1/{len(bitstring)*8}\t\t" + f"attempt {0}\n"
        )
        print(base_output + initial_prompt)
        for index, bit in enumerate(iter_bits(bitstring)):
            (next_prompt, tries) = get_next_tokens(
                llm,
                current_prompt,
                bit,
                h,
                chars_per_bit,
                message=f"bit {index + 1}/{len(bitstring) * 8}\t\t",
            )
            current_prompt += next_prompt
            total_tries += tries

    end_time = timeit.default_timer()

    print(f"Elapsed: {round((end_time - start_time) * 1e6, 3)} µs")
    print(f"Elapsed: {round((end_time - start_time) * 1e3, 3)} ms")
    print(f"Elapsed per next token gen: {round((end_time - start_time) * 1e3/total_tries, 3)} ms")

    print("total_tries: ", total_tries)
    print("len(bitstring): ", len(bitstring))
    print("avg try: ", total_tries/(len(bitstring)*8))
    return current_prompt

def decode_bitstring(
    encoded_message: str, initial_prompt: str, h, num_bits: int, chars_per_bit=16
) -> str:
    """
    Decodes a bitstring from the encoded LLM output.

    Args:
        encoded_message (str): The full message output from the LLM after encoding.
        initial_prompt (str): The prompt that was used to generate the message.
        h: The hash function used to encode/decode bits.
        num_bits (int): The number of bits to decode.
        chars_per_bit (int): The number of characters per encoded bit.

    Returns:
        str: The decoded message as a string.
    """
    # Remove the initial prompt from the encoded message
    if not encoded_message.startswith(initial_prompt):
        raise ValueError("Encoded message does not start with the initial prompt.")
    data = encoded_message[len(initial_prompt):]
    bits = []

    for i in range(num_bits):
        chunk = data[:chars_per_bit]
        bit = first_bit_of_hash(h, chunk.encode())
        bits.append(bit)
        data = data[chars_per_bit:]

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
    return decoded_bytes.rstrip(b"\x00").decode(errors="replace")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="The Alvarez hypothesis posits that the mass extinction of the dinosaurs and many other living things during the Cretaceous-Paleogene extinction event",
    )

    args = parser.parse_args()

    h = hashlib.sha256

    llm = Llama(model_path=args.model_path, verbose=False, n_ctx=0, device="cuda:0")

    #bitstring = b"hello"
    bitstring = b"abcdefg"

    print("message to encode: " + str(bitstring))
    print("length: ", len(bitstring))

    chars_per_bit_i = 32
    print("chars_per_bit_i: ", chars_per_bit_i)

    try:
        # Code that might raise a SystemError
        message = encode_bitstring(llm, args.prompt, bitstring, h, chars_per_bit_i)
    except SystemError as e:
        # Handle the SystemError
        message = encode_bitstring(llm, args.prompt, bitstring, h, chars_per_bit_i)


    print("~~~~~~~\nMessage:\n~~~~~~~\n", message)

    # --- Decode the message back ---
    num_bits = len(bitstring) * 8
    decoded = decode_bitstring(
        message,
        args.prompt,
        h,
        num_bits,
        chars_per_bit_i
    )
    print("~~~~~~~\nDecoded message:\n~~~~~~~\n", decoded)