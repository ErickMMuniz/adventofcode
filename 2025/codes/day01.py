from functools import reduce

compose = lambda *funcs: reduce(lambda f, g: lambda x: g(f(x)), funcs)

INPUT_01: str = "input/day01_01.txt"
INIT_DIAL: int = 50

StateWithCount = tuple[str, int, int]
initial_state_count: StateWithCount = ("S", INIT_DIAL, 0)


def strip_and_split(text: str) -> list[str]:
    return text.strip().split("\n")


def split_letter_and_number(data: list[str]) -> list[StateWithCount]:

    def split_one(row: str) -> StateWithCount:
        return row[0], int(row[1:]), 0

    return map(split_one, data)


def calculate_dial(states: list[StateWithCount]) -> StateWithCount:

    def function_(s_1: StateWithCount, s_2: StateWithCount) -> StateWithCount:
        state, value, count = s_1
        direction, delta, _ = s_2

        match direction:
            case "R":
                value = (value + delta) % 100
            case "L":
                value = (value - delta + 100) % 100

        if value == 0:
            count += 1

        return state, value, count

    return reduce(function_, states, initial_state_count)


if __name__ == "__main__":
    with open(INPUT_01) as f:
        text = f.read()
        print(compose(strip_and_split, split_letter_and_number, calculate_dial)(text))
        s = strip_and_split(text)
