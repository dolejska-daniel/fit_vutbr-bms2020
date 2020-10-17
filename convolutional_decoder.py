import logging
from collections import deque

from utils import *
from queue import PriorityQueue

logger = logging.getLogger("decoder")


class ConvolutionalDecoder(object):

    def __init__(self, stage_count: int, feedback_masks: typing.List[int]):
        self.stage_count = stage_count
        self.feedback_masks = feedback_masks

        # generate all encoder states
        self._states = all_states(stage_count)
        # generate table of encoder results in all possible states with all possible inputs
        self._emissions = emission_table(stage_count, feedback_masks)
        # generate table of encoder state transitions between all possible states with all possible inputs
        self._transitions = transition_table(stage_count)

        # initialize priority queue which will contain unprocessed states
        # going over encoder transitions and reverse-generating encoded data
        self._unprocessed_states = PriorityQueue()
        # initialize dictionary which will contain currently lowest cost of state per iteration
        self._lowest_iteration_cost = defaultdict(dict)

    def _initialize_decoder(self):
        self._unprocessed_states = PriorityQueue()
        self._lowest_iteration_cost = defaultdict(dict)

    def _create_unprocessed_state(self, state_cost: int, current_state: str, data_out: typing.List[int], data_in: str):
        self._unprocessed_states.put((state_cost, current_state, data_out, data_in))

    def _get_unprocessed_state(self) -> typing.Tuple[int, str, typing.List[int], str]:
        return self._unprocessed_states.get()

    def _get_best_path_cost(self, current_iteration: int, current_state: str) -> typing.Union[int, None]:
        if current_state not in self._lowest_iteration_cost[current_iteration]:
            return None

        return self._lowest_iteration_cost[current_iteration][current_state]

    def _does_better_path_exist(self, current_iteration: int, current_cost: int, current_state: str) -> bool:
        current_best_cost = self._get_best_path_cost(current_iteration, current_state)
        return current_best_cost <= current_cost if current_best_cost is not None else False

    def _register_best_path_cost(self, current_iteration: int, current_cost: int, current_state: str):
        self._lowest_iteration_cost[current_iteration][current_state] = current_cost

    @classmethod
    def int_binary_to_str(cls, data_in: typing.List[int]) -> str:
        if len(data_in) < 8:
            # there is not enough bytes for whole byte
            return ""

        # list of decoded ASCII characters
        result = []
        # value of current byte
        byte_value = 0
        # for each bit in decoded binary sequence
        for bit_index, bit in enumerate(data_in):
            # does current bit belong to another byte?
            if bit_index % 8 == 0 and bit_index > 0:
                # convert current byte value to ASCII character
                result.append(chr(byte_value))
                # reset byte value
                byte_value = 0
                if bit_index + 8 > len(data_in):
                    # there is not enough bytes for whole byte left
                    break

            # calculate value of current bit and sum it with current byte value
            byte_value += pow(2, bit_index % 8) * bit

        # characters were decoded in reverse order
        result.reverse()
        # create final string and return as result
        return "".join(result)

    def decode(self, data_in: str, max_result_count: int = 3) -> typing.List[typing.Tuple[int, str]]:
        # re-initialize inner decoder state
        self._initialize_decoder()
        # create initial state to process
        self._create_unprocessed_state(0, state_to_str(self._states[0]), [], data_in)

        # initialize list of results
        results = []
        # this defines the amount of
        observation_offset = len(self.feedback_masks)

        # while there are unprocessed states
        while not self._unprocessed_states.empty() and len(results) < max_result_count:
            # get currently best unprocessed state
            current_cost, current_state, current_solution, observations_left = self._get_unprocessed_state()
            # calculate current iteration identifier
            current_iteration = len(current_solution)
            logger.debug("processing path from %s (iter=%2d, cost=%2d)", current_state, current_iteration, current_cost)

            if self._does_better_path_exist(current_iteration, current_cost, current_state):
                # there already is solution path with better cost
                # and since branches from a signle state cannot diverge
                # lets skip this worse path
                logger.debug("skipping current path (cost=%2d), better path leading to current state exists (cost=%2d)",
                             current_cost, self._get_best_path_cost(current_iteration, current_state))
                continue

            # since there is no lower cost store current cost
            # as the currently lowest for given iteration and state
            self._register_best_path_cost(current_iteration, current_cost, current_state)
            if not observations_left:
                # there is no input left - this is the final state and possible solution
                logger.debug("possible solution found, saving result %s (iter=%2d, cost=%2d)",
                             current_solution, current_iteration, current_cost)
                results.append((current_cost, self.int_binary_to_str(current_solution)))
                continue

            # select observation in current state
            current_observation = observations_left[-observation_offset:]
            # for each possible branch from current state do
            for possible_bit_value in [0, 1]:
                # what would be the encoder output
                # if it were in `current_state` and received `possible_bit_value`
                transition_observation = state_to_str(self._emissions[current_state][possible_bit_value])
                # what is the cost of this transition - how is the possible encoder output
                # different from what I've received
                transition_cost = hamming_distance(transition_observation, current_observation)
                # to which state would encoder transition in case of `possible_bit_value` input
                next_state = self._transitions[current_state][possible_bit_value]

                # create new state to process
                self._create_unprocessed_state(
                    # accumulate transition costs
                    transition_cost + current_cost,
                    # current state will be state after transition
                    next_state,
                    # add transition "input" value to path solution
                    current_solution.copy() + [possible_bit_value],
                    # remove observation in current state from next state
                    observations_left[:-observation_offset],
                )

        return results
