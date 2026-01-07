import argparse
import itertools

from tqdm import tqdm


def left_pad(s: str, target_len: int) -> str:
    while len(s) < target_len:
        s = "0" + s
    return s


def get_all_strings(num_bits: int) -> list[str]:
    all_strings = []

    for i in tqdm(range(2 ** (num_bits))):
        base_s = bin(i)[2:]
        all_strings.append(left_pad(base_s, num_bits))
    return all_strings


def count_appearences(strs: list[str], flags: list[str]) -> int:
    count = 0
    for s in strs:
        for flag in flags:
            if flag in s:
                count += 1
                break
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", type=int, help="The length of the sampled string", default=8
    )
    parser.add_argument("-q", type=int, help="The flag length", default=2)
    parser.add_argument("-z", type=int, help="The number of flags", default=1)
    args = parser.parse_args()

    possible_strings = get_all_strings(args.p)

    best = (-1, None)
    others = []
    for combo in tqdm(itertools.combinations(get_all_strings(args.q), args.z)):
        apps = count_appearences(possible_strings, combo)

        if apps > best[0]:
            best = (apps, combo)
            others = []

        elif apps == best[0]:
            others.append(combo)
    print(best)
    for c in others:
        print(c)

    print(best[0] / 2 ** (args.p))
