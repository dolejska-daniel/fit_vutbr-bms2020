import typing
from collections import defaultdict
from itertools import product

State = typing.NewType("State", typing.List[int])


def state_to_str(state: State) -> str:
    return "".join(map(lambda x: str(x), state))


def str_to_state(state: str) -> State:
    return State(list(map(lambda x: int(x), state)))


def all_states(stage_count: int) -> typing.List[State]:
    if stage_count < 1:
        raise RuntimeError("Cannot count states. Invalid stage count.")

    return [
        State(list(map(lambda x: int(x), state)))
        for state in product("01", repeat=stage_count)
    ]


def hamming_distance(param1: str, param2: str):
    distance = 0
    for char1, char2 in zip(param1, param2):
        if char1 != char2:
            distance += 1

    return distance + abs(len(param1) - len(param2))


def emission_table(stage_count: int, feedback_masks: typing.List[int]) \
        -> typing.Dict[str, typing.Dict[int, typing.List[int]]]:
    from convolution_filter import ConvolutionFilter
    table = defaultdict(dict)
    states = all_states(stage_count)
    _filter = ConvolutionFilter(stage_count, feedback_masks)

    for source_state in states:
        _filter.initialize(source_state)
        source_state_key = state_to_str(source_state)

        table[source_state_key][0] = _filter.output_for(0)
        table[source_state_key][1] = _filter.output_for(1)

    return table


def transition_table(stage_count: int) -> typing.Dict[str, typing.Dict[int, str]]:
    table = defaultdict(dict)
    states = all_states(stage_count)

    for source_state in states:
        source_state_key = state_to_str(source_state)
        table[source_state_key][0] = state_to_str([0] + source_state[:-1])
        table[source_state_key][1] = state_to_str([1] + source_state[:-1])

    return table
