import argparse
import hashlib
import random

import blessed
from llama_cpp import Llama

import timeit
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

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
        if increment > 100:
            raise SystemError("Giving up after trying to generate next token with desired hash.")            

        modified_base = base_output + f"attempt {increment + 1}\n"

        result = llm(prompt, max_tokens=num_chars, seed=seed + increment,
                     #min_p=0.05,  # Minimum probability threshold
                     temperature=1.3,  # Add some randomness
                     repeat_penalty=1.2  # Discourage repetition
                     )
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

    ##############################

    # Alice generates her private key
    alice_private_key = x25519.X25519PrivateKey.generate()
    alice_public_key = alice_private_key.public_key()

    # Bob generates his private key
    bob_private_key = x25519.X25519PrivateKey.generate()
    bob_public_key = bob_private_key.public_key()

    # Alice computes the shared secret using Bob's public key
    alice_shared_secret = alice_private_key.exchange(bob_public_key)

    # Bob computes the shared secret using Alice's public key
    bob_shared_secret = bob_private_key.exchange(alice_public_key)

    # Both shared secrets are identical
    assert alice_shared_secret == bob_shared_secret

    # Derive key using HKDF
    kdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'prf-key',
    )
    prf_key = kdf.derive(alice_shared_secret)

    print(f"PRF Key: {prf_key.hex()}")

    # PRF evaluation at input 0 using HMAC-SHA256
    # PRF(key, input) = HMAC-SHA256(key, input)
    prf = hmac.HMAC(prf_key, hashes.SHA256())
    prf.update(b'\x00')  # Input = 0 as a single byte
    prf_output = prf.finalize()

    print(f"PRF(key, 0): {prf_output.hex()}")
    print(f"PRF output length: {len(prf_output)} bytes")

    concatenated_key_prf_out = b''.join([prf_key, prf_output])

    # Extract prf_key (first 32 bytes)
    extracted_prf_key = concatenated_key_prf_out[:32]

    # Extract prf_output (next 32 bytes)
    extracted_prf_output = concatenated_key_prf_out[32:64]
    # Or simply: concatenated[32:]

    # Verify they match the originals
    assert extracted_prf_key == prf_key
    assert extracted_prf_output == prf_output
    prf_i = hmac.HMAC(extracted_prf_key, hashes.SHA256())
    prf_i.update(b'\x00')  # Input = 0 as a single byte
    prf_output_i = prf_i.finalize()
    assert extracted_prf_output == prf_output_i

    ###########################################

    args = parser.parse_args()

    h = hashlib.sha256

    #bitstring = b"hello"
    bitstring = b"abcdefg"
    print("bitstring type: ", type(bitstring))

    # replacing test bitstring with the key and prf eval of 0
    bitstring = concatenated_key_prf_out
    print("new bitstring type: ", type(bitstring))

    print("message to encode: " + str(bitstring))
    print("length: ", len(bitstring))

    chars_per_bit_i = 32
    print("chars_per_bit_i: ", chars_per_bit_i)

    llm = Llama(model_path=args.model_path, verbose=False, n_ctx=0, device="cuda:0")

    max_retries = 10
    for attempt in range(max_retries):
        try:
            message = encode_bitstring(llm, args.prompt, bitstring, h, chars_per_bit_i)
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
        message,
        args.prompt,
        h,
        num_bits,
        chars_per_bit_i
    )
    print("~~~~~~~\nDecoded message:\n~~~~~~~\n", decoded)

    assert decoded_bytes == bitstring