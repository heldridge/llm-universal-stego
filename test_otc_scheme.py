import unittest

from otc_scheme import keygen, otc_encode, otc_decode


class OTCSchemeTests(unittest.TestCase):
    def test_round_trip(self):
        sk = keygen(256)
        plaintext = b"hello otc"
        encoded = otc_encode(sk, plaintext, mac_len_bytes=16, rs_nsym=16)
        decoded = otc_decode(sk, encoded, mac_len_bytes=16, rs_nsym=16)
        self.assertEqual(decoded, plaintext)

    def test_wrong_key_rejected(self):
        sk = keygen(256)
        wrong_sk = keygen(256)
        plaintext = b"secret"
        encoded = otc_encode(sk, plaintext, mac_len_bytes=16, rs_nsym=16)
        decoded = otc_decode(wrong_sk, encoded, mac_len_bytes=16, rs_nsym=16)
        self.assertIsNone(decoded)

    def test_tampering_rejected(self):
        sk = keygen(256)
        plaintext = b"tamper-check"
        encoded = bytearray(otc_encode(sk, plaintext, mac_len_bytes=16, rs_nsym=16))
        for i in range(min(9, len(encoded))):
            encoded[i] ^= 0x01
        decoded = otc_decode(sk, bytes(encoded), mac_len_bytes=16, rs_nsym=16)
        self.assertIsNone(decoded)

    def test_configurable_key_bits(self):
        self.assertEqual(len(keygen(128)), 16)
        self.assertEqual(len(keygen(384)), 48)
        with self.assertRaises(ValueError):
            keygen(130)

    def test_configurable_reed_solomon(self):
        sk = keygen(256)
        plaintext = b"rs configurable"
        encoded = otc_encode(sk, plaintext, mac_len_bytes=12, rs_nsym=24, length_field_bytes=2)
        decoded = otc_decode(sk, encoded, mac_len_bytes=12, rs_nsym=24, length_field_bytes=2)
        self.assertEqual(decoded, plaintext)


if __name__ == "__main__":
    unittest.main()
