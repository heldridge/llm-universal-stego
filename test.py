import argparse
import hashlib
import random

import blessed
from llama_cpp import Llama

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
    llm: Llama, prompt: str, target_bit: int, h, num_chars=16, message=""
) -> str:
    seed = random.randint(0, 2**64)
    increment = 0
    base_output = term.home + term.clear + message
    while True:

        modified_base = base_output + f"attempt {increment + 1}\n"

        result = llm(prompt, max_tokens=num_chars, seed=seed + increment)
        output = result["choices"][0]["text"]
        if len(output) >= num_chars:
            final_str = output[:num_chars]

            hash_works = first_bit_of_hash(h, final_str.encode()) == target_bit

            if hash_works:
                print(modified_base + prompt + term.green(final_str))
                return final_str
            else:
                print(modified_base + prompt + term.red(final_str))

        increment += 1


def encode_bitstring(
    llm: Llama, initial_prompt: str, bitstring: bytes, h, chars_per_bit=16
) -> str:
    current_prompt = initial_prompt

    with term.fullscreen(), term.cbreak():
        base_output = (
            term.home + term.clear + f"bit 1/{len(bitstring)*8}\t\t" + f"attempt {0}\n"
        )
        print(base_output + initial_prompt)
        for index, bit in enumerate(iter_bits(bitstring)):
            current_prompt += get_next_tokens(
                llm,
                current_prompt,
                bit,
                h,
                chars_per_bit,
                message=f"bit {index + 1}/{len(bitstring) * 8}\t\t",
            )

    return current_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="The Alvarez hypothesis posits that the mass extinction of the dinosaurs and many other living things during the Cretaceous-Paleogene extinction event",
    )
    parser.add_argument(
        "--chars-per-bit",
        "-c",
        type=int,
        default=32,
        help="The number of characters used to encode each bit",
    )
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        default="hello",
        help="The message to encode in the LLM output",
    )

    args = parser.parse_args()

    h = hashlib.sha256

    llm = Llama(model_path=args.model_path, verbose=False, n_ctx=0)

    bitstring = args.message.encode()

    message = encode_bitstring(llm, args.prompt, bitstring, h, args.chars_per_bit)
    print("~~~~~~~\nMessage:\n~~~~~~~\n", message)
