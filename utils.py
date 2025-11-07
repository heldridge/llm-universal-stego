import noisy_elligator as ne

def gen_elligator_bitstring(num_flag_bytes = 8):
    params = ne.Parameters(b"waterlog", b"0"*num_flag_bytes, 32)
    _, spk = ne.gen_server()
    _, m = ne.gen_client_message(params, [spk])
    return m
