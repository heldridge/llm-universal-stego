import argparse
import hashlib
import random
import math
import os
import copy

import blake3
from numpy.ma.extras import average

import noisy_elligator as ne
from test_embedd_decode import encode_bitstring, decode_bitstring_to_bits

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from colorama import init, Fore, Style

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, BitsAndBytesConfig

import timeit
import utils

tokens_per_bit_i = 3
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

def random_bits(length: int) -> list[int]:
    """Generate a list of random bits (0s or 1s)."""
    return random. choices([0, 1], k=length)

def bits_to_bytes(bits: list[int]) -> bytes:
    """
    Convert a list of bits (0s and 1s) back to bytes.

    Args:
        bits: List of integers (0 or 1) representing bits

    Returns:
        bytes: The reconstructed byte string
    """
    result = bytearray()

    # Process 8 bits at a time
    for i in range(0, len(bits), 8):
        # Get the next 8 bits (or remaining bits if less than 8)
        byte_bits = bits[i:i + 8]

        # Convert bits to byte value
        byte_value = 0
        for bit in byte_bits:
            byte_value = (byte_value << 1) | bit

        result.append(byte_value)

    return bytes(result)


def check_if_pre_shared_key(input_bytes: bytes, pre_shared_key: bytes) -> bool:
    """
    Check if two byte strings are equal in constant time.
    Expects inputs to be the output of bits_to_bytes().
    """
    if len(input_bytes) != len(pre_shared_key):
        return False

    result = 0
    # Iterating over bytes yields integers (0-255)
    for x, y in zip(input_bytes, pre_shared_key):
        # XOR checks for difference, OR accumulates the result
        result |= x ^ y

    return result == 0

def find_sync_code_with_data(bits: list[int], sync_code: list[int], output_length: int) -> list[tuple[int, list[int]]]:
    """
    Find all indices where the sync code appears and extract data after it.

    Args:
        bits: The larger list of bits (0s and 1s) to search in
        sync_code: The sync code pattern (list of 0s and 1s) to search for
        output_length: Number of bits to extract after each sync code

    Returns:
        A list of tuples (index, data) where:
        - index is the starting position of the sync code
        - data is a list of output_length bits following the sync code

    Example:
        >>> bits = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
        >>> sync_code = [0, 1]
        >>> find_sync_code_with_data(bits, sync_code, 3)
        [(0, [0, 1, 1]), (2, [1, 1, 0]), (6, [1, 1, 1])]
    """
    # Convert to strings for fast pattern matching
    bits_str = ''.join(map(str, bits))
    sync_str = ''.join(map(str, sync_code))

    sync_len = len(sync_code)
    total_len = len(bits)
    results = []
    start = 0

    while True:
        # Find next occurrence of sync code
        pos = bits_str.find(sync_str, start)
        if pos == -1:
            break

        # Calculate where data starts and ends
        data_start = pos + sync_len
        data_end = data_start + output_length

        # Extract data only if there's enough space
        if data_end <= total_len:
            data = bits[data_start:data_end]
            results.append((pos, data))

        # Move to next position
        start = pos + 1

    return results

SYMMETRIC_SETTING = False

num_flag_bytes = 2;
params = ne.Parameters(b"waterlog", b"0"*num_flag_bytes, 32)
ssk, spk = ne.gen_server()
shared_secrets_client, msg = ne.gen_client_message(params, [spk])

msg_bits = list(iter_bits(msg));
#print(f"msg_bits (len = {len(msg_bits)}) is: ", msg_bits)

def_prompt = "The Alvarez hypothesis posits that the mass extinction of the dinosaurs and many other living things during the Cretaceous-Paleogene extinction event"
h = blake3.blake3

### benchmark the time to build this list with a static length for bits
runs_inner = 1
runs_outer = 1
total_runs = runs_inner*runs_outer

# 500, 1500, 2000, 2500, 3000
#total_bits = 500
# 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000
total_bits_list = [500]
watermark_len = 272
msg_bits = msg_bits[:watermark_len]
assert len(msg_bits) == watermark_len
pre_shared_key_bytes = bits_to_bytes(msg_bits)


for total_bits in total_bits_list:
    # 4, 5, 6, 7, 8, 10, 12 || 4, 6, 8, 10, 12
    sync_code_len_list = [2] #[2, 4, 6, 8, 10, 12]
    for sync_code_len in sync_code_len_list:
    #sync_code_len = 8
        total_runtime = 0;
        for i in range(runs_outer):
            write_index = random.randint(0, total_bits-watermark_len-sync_code_len)
            print("write_index is: ", write_index)
            sync_code = random_bits(sync_code_len)
            rand_start_bits = random_bits(write_index)
            rand_end_bits = random_bits(total_bits-write_index-watermark_len-sync_code_len)
            bits = rand_start_bits + sync_code + msg_bits + rand_end_bits
            assert len(bits) == total_bits

            # Generate string from bits
            # Reconstruct Bytes from Bits
            full_msg_bytes = bytearray()
            for b in range(0, len(bits), 8):
                byte = 0
                for i in range(8):
                    if b + i < len(bits):
                        byte = (byte << 1) | bits[b + i]
                    else:
                        byte = (byte << 1)
                full_msg_bytes.append(byte)

            full_msg_string = full_msg_bytes.rstrip(b"\x00").decode(errors="replace")

            correct_result = -1;

            model_path = "mistralai/Mistral-7B-Instruct-v0.3"
            load_kwargs = {"device_map": "auto", "dtype": torch.float16}
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = quantization_config
            max_memory_mapping = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    max_memory_mapping[i] = "8GiB"
            load_kwargs["max_memory"] = {0: "8GiB"}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            max_retries = 20
            message = None

            for attempt in range(max_retries):
                try:
                    (message, _) = encode_bitstring(
                        model,
                        tokenizer,
                        def_prompt,
                        full_msg_bytes,
                        h,
                        tokens_per_bit_i,
                        1.0,
                        target_output_tokens=total_bits,
                        quiet = True
                    )
                    break
                except SystemError as e:
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt == max_retries - 1:
                        raise


            start_time = timeit.default_timer()
            ############################################################

            for i in range(runs_inner):
                # decode_bitstring_to_bits
                recovered_bits = decode_bitstring_to_bits(
                    tokenizer,
                    def_prompt,
                    h,
                    message,
                    tokens_per_bit_i,
                    num_bits_to_decode=total_bits,
                )
                results = find_sync_code_with_data(recovered_bits, sync_code, watermark_len)

                for (idx, result) in enumerate(results):
                    recovered_msg = bits_to_bytes(result[1])
                    try:
                        if SYMMETRIC_SETTING:
                            found = check_if_pre_shared_key(recovered_msg,  pre_shared_key_bytes)
                            if not found:
                                raise Exception("Not the pre-shared key")
                        else:
                            shared_secret_server = ne.process_client_message(params, ssk, recovered_msg)
                    except:
                        2  # print("decrypt fail")
                    else:
                        correct_result = result[0]

            end_time = timeit.default_timer()
            total_runtime += round((end_time - start_time), 15)

        print(f"Total Elapsed: {total_runtime} secs, correct_start_index: {correct_result}")
        average_sync_code_time = total_runtime/total_runs;
        with open("find_sync_bench.txt", "a+") as f:
            print(f"Bit Length: {total_bits} Sync Code Length: {sync_code_len} Total Runs: {total_runs} Average Runtime: {average_sync_code_time}\n")
            f.write(f"Bit Length: {total_bits} Sync Code Length: {sync_code_len} Total Runs: {total_runs} Average Runtime: {average_sync_code_time}\n")
