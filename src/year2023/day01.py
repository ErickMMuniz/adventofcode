import re
import functools
import os
from typing import List, Text
import numpy as np

# AoC => AdventOfCode
DAY_AoC = os.path.basename(__file__).split(".")[0]
INPUT_AoC = os.path.join(
    os.getcwd(), "src", "year2023", "inputs", "{}.txt".format(DAY_AoC)
)
INPUTS = np.loadtxt(INPUT_AoC, dtype=str)

VALID_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def extract_head_and_tail_digist(text: Text) -> int:
    matches: List[int] = re.findall("\d", text)
    head, last = matches[0], matches[-1]
    return int("".join([head, last]))


def to_number(maybe_int: Text) -> int:
    if maybe_int in VALID_NUMBERS:
        return str(VALID_NUMBERS[maybe_int])
    else:
        return maybe_int


def extract_head_and_last_digit_with_words(text: Text) -> int:
    pattern = "({}|\d)".format("|".join(VALID_NUMBERS.keys()))
    overlaping_pattern = "(?={})".format(pattern)

    matches: List[int] = re.findall(overlaping_pattern, text)

    head, last = matches[0], matches[-1]

    return int("".join([to_number(head), to_number(last)]))


def first_star() -> bool:
    f_1 = np.vectorize(extract_head_and_tail_digist)
    result = f_1(INPUTS).sum()
    return 54708 == result


def second_star() -> bool:
    f_2 = np.vectorize(extract_head_and_last_digit_with_words)
    result = f_2(INPUTS).sum()
    return 54087 == result
