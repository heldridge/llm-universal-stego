"""OTC-style authenticated encoding helpers for byte payloads."""

from __future__ import annotations

import hashlib
import hmac
import secrets


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("Byte strings must have equal length for XOR")
    return bytes(a ^ b for a, b in zip(left, right))


def _require_nonempty_bytes(value: bytes, label: str) -> None:
    if not isinstance(value, (bytes, bytearray)) or len(value) == 0:
        raise ValueError(f"{label} must be non-empty bytes")


def _validate_key_bits(key_bits: int) -> int:
    if key_bits < 128 or key_bits % 8 != 0:
        raise ValueError("key_bits must be >= 128 and divisible by 8")
    return key_bits


def _validate_mac_len(mac_len_bytes: int) -> int:
    if mac_len_bytes < 8 or mac_len_bytes > hashlib.sha256().digest_size:
        raise ValueError("mac_len_bytes must be between 8 and 32")
    return mac_len_bytes


def _validate_rs_nsym(rs_nsym: int) -> int:
    if rs_nsym < 2 or rs_nsym > 128:
        raise ValueError("rs_nsym must be between 2 and 128")
    return rs_nsym


def _get_rs_codec(nsym: int):
    try:
        from reedsolo import RSCodec
    except ImportError as exc:
        raise ImportError(
            "reedsolo is required for OTC mode. Install it with `pip install reedsolo`."
        ) from exc
    return RSCodec(nsym)


def keygen(key_bits: int = 256) -> bytes:
    """Generate a cryptographically random secret key of `key_bits` length."""
    key_bits = _validate_key_bits(key_bits)
    return secrets.token_bytes(key_bits // 8)


def _prf_stream(secret_key: bytes, domain: int, length: int) -> bytes:
    if domain not in (0, 1):
        raise ValueError("PRF domain must be 0 or 1")
    _require_nonempty_bytes(secret_key, "secret_key")
    if length < 0:
        raise ValueError("length must be non-negative")
    output = bytearray()
    counter = 0
    while len(output) < length:
        block = hmac.new(
            bytes(secret_key),
            bytes([domain]) + counter.to_bytes(4, "big"),
            hashlib.sha256,
        ).digest()
        output.extend(block)
        counter += 1
    return bytes(output[:length])


def otc_encode(
    secret_key: bytes,
    plaintext: bytes,
    mac_len_bytes: int = 16,
    rs_nsym: int = 16,
    length_field_bytes: int = 2,
) -> bytes:
    """Encode plaintext with OTC wrapper: MAC + RS + XOR stream mask."""
    _require_nonempty_bytes(secret_key, "secret_key")
    if not isinstance(plaintext, (bytes, bytearray)):
        raise ValueError("plaintext must be bytes")
    mac_len_bytes = _validate_mac_len(mac_len_bytes)
    rs_nsym = _validate_rs_nsym(rs_nsym)
    if length_field_bytes not in (2, 4):
        raise ValueError("length_field_bytes must be 2 or 4")

    plaintext = bytes(plaintext)
    max_plaintext_len = (1 << (8 * length_field_bytes)) - 1
    if len(plaintext) > max_plaintext_len:
        raise ValueError("plaintext too long for configured length_field_bytes")

    max_rs_payload = 255 - rs_nsym
    payload_len = length_field_bytes + len(plaintext) + mac_len_bytes
    if payload_len > max_rs_payload:
        raise ValueError(
            f"payload too large for rs_nsym={rs_nsym}; max payload bytes is {max_rs_payload}"
        )

    mac_key = _prf_stream(bytes(secret_key), 1, hashlib.sha256().digest_size)
    tag = hmac.new(mac_key, plaintext, hashlib.sha256).digest()[:mac_len_bytes]
    payload = len(plaintext).to_bytes(length_field_bytes, "big") + plaintext + tag

    rs_codec = _get_rs_codec(rs_nsym)
    rs_encoded = bytes(rs_codec.encode(payload))
    mask = _prf_stream(bytes(secret_key), 0, len(rs_encoded))
    return _xor_bytes(rs_encoded, mask)


def otc_decode(
    secret_key: bytes,
    ciphertext: bytes,
    mac_len_bytes: int = 16,
    rs_nsym: int = 16,
    length_field_bytes: int = 2,
) -> bytes | None:
    """Decode OTC payload; returns plaintext or None on failure."""
    _require_nonempty_bytes(secret_key, "secret_key")
    if not isinstance(ciphertext, (bytes, bytearray)):
        raise ValueError("ciphertext must be bytes")
    mac_len_bytes = _validate_mac_len(mac_len_bytes)
    rs_nsym = _validate_rs_nsym(rs_nsym)
    if length_field_bytes not in (2, 4):
        raise ValueError("length_field_bytes must be 2 or 4")

    ciphertext = bytes(ciphertext)
    if len(ciphertext) <= rs_nsym:
        return None

    mask = _prf_stream(bytes(secret_key), 0, len(ciphertext))
    rs_encoded = _xor_bytes(ciphertext, mask)
    rs_codec = _get_rs_codec(rs_nsym)

    try:
        rs_decoded = rs_codec.decode(rs_encoded)
    except Exception:
        return None

    payload = bytes(rs_decoded[0] if isinstance(rs_decoded, tuple) else rs_decoded)
    min_payload_len = length_field_bytes + mac_len_bytes
    if len(payload) < min_payload_len:
        return None

    plaintext_len = int.from_bytes(payload[:length_field_bytes], "big")
    expected_payload_len = length_field_bytes + plaintext_len + mac_len_bytes
    if expected_payload_len != len(payload):
        return None

    plaintext = payload[length_field_bytes:length_field_bytes + plaintext_len]
    tag = payload[length_field_bytes + plaintext_len:]
    mac_key = _prf_stream(bytes(secret_key), 1, hashlib.sha256().digest_size)
    expected_tag = hmac.new(mac_key, plaintext, hashlib.sha256).digest()[:mac_len_bytes]
    if not hmac.compare_digest(tag, expected_tag):
        return None
    return plaintext
