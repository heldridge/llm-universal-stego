import collections
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TypeVar

import monocypher

type ServerPublicKey = bytes
type ServerSecretKey = bytes
type SharedSecret = bytes
type HiddenKey = bytes
type Nonce = bytes
type ClientMessage = bytes
type Ciphertext = bytes


class DecodeFlagFailure(Exception):
    pass


class StegoMessageDetectionFailure(Exception):
    pass


T = TypeVar("T")


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def _sliding_window(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    iterator = iter(iterable)
    window = collections.deque(itertools.islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


def _sliding_bytes_window(byte_string: bytes, n: int) -> Iterator[bytes]:
    it = _sliding_window(byte_string, n)
    for x in it:
        yield bytes(x)


@dataclass
class Parameters:
    nonce: bytes
    flag_message: bytes
    hidden_key_length: int

    def __post_init__(self):
        if len(self.nonce) not in [8, 24]:
            raise ValueError("Nonce must be either 8 or 24 bytes")


"""
Run by Server
"""


def gen_server() -> tuple[ServerSecretKey, ServerPublicKey]:
    return monocypher.generate_key_exchange_key_pair()


def _compute_shared_secret_server(
    server_sk: ServerSecretKey, client_hk: HiddenKey
) -> SharedSecret:
    client_pk = monocypher.elligator_map(client_hk)
    shared_secret = monocypher.key_exchange(server_sk, client_pk)
    return shared_secret


def _shared_secret_is_correct(
    params: Parameters, shared_secret: SharedSecret, encrypted_flag: Ciphertext
) -> bool:
    if (
        monocypher.chacha20(shared_secret, params.nonce, encrypted_flag)
        == params.flag_message
    ):
        return True
    else:
        return False


def _attempt_shared_secret_reconstruction(
    params: Parameters,
    server_sk: ServerSecretKey,
    client_hk_candidate: HiddenKey,
    encrypted_flag_candidate: Ciphertext,
) -> SharedSecret:
    shared_secret = _compute_shared_secret_server(server_sk, client_hk_candidate)
    if not _shared_secret_is_correct(params, shared_secret, encrypted_flag_candidate):
        raise DecodeFlagFailure
    return shared_secret


def process_client_message(
    params: Parameters, server_sk: ServerSecretKey, client_message: ClientMessage
) -> SharedSecret:
    flag_length = len(params.flag_message)

    shared_secret = None
    for candidate_bytes in _sliding_bytes_window(
        client_message, flag_length + params.hidden_key_length
    ):
        encrypted_flag_candidate = candidate_bytes[:flag_length]
        client_hk_candidate = candidate_bytes[flag_length:]

        try:
            shared_secret = _attempt_shared_secret_reconstruction(
                params, server_sk, client_hk_candidate, encrypted_flag_candidate
            )
        except DecodeFlagFailure:
            continue

    if shared_secret is None:
        raise StegoMessageDetectionFailure

    return shared_secret


"""
Run by Client
"""


def _gen_client_message_parts(
    params: Parameters,
    server_pk: ServerPublicKey,
) -> tuple[SharedSecret, HiddenKey, Ciphertext]:
    client_hk, client_sk = monocypher.elligator_key_pair()
    shared_secret = monocypher.key_exchange(client_sk, server_pk)
    encrypted_flag = monocypher.chacha20(
        shared_secret, params.nonce, params.flag_message
    )
    return shared_secret, client_hk, encrypted_flag


def _encode_client_message(
    client_hks: list[HiddenKey], encrypted_flags: list[Ciphertext]
) -> ClientMessage:

    msg: bytes = b""

    for client_hk, encrypted_flag in zip(client_hks, encrypted_flags):
        msg += encrypted_flag + client_hk

    return msg


def gen_client_message(
    params: Parameters,
    server_pks: list[ServerPublicKey],
) -> tuple[list[SharedSecret], ClientMessage]:

    shared_secrets = []
    client_hks = []
    encrypted_flags = []
    for server_pk in server_pks:
        shared_secret, client_hk, encrypted_flag = _gen_client_message_parts(
            params, server_pk
        )

        shared_secrets.append(shared_secret)
        client_hks.append(client_hk)
        encrypted_flags.append(encrypted_flag)

    return shared_secrets, _encode_client_message(client_hks, encrypted_flags)
