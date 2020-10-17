import logging
from utils import *
from queue import PriorityQueue

logger = logging.getLogger("decoder")


def viterbi_own(memory_stage_count, feedback_masks, data_in):
    states = all_states(memory_stage_count)
    emissions = emission_table(memory_stage_count, feedback_masks)
    transitions = transition_table(memory_stage_count)
    observation_offset = len(feedback_masks)

    queue = PriorityQueue()
    queue.put((0, state_to_str(states[0]), [], data_in))

    best_iteration_cost = defaultdict(dict)
    finished_paths = []
    while not queue.empty():
        current_cost, current_state, current_solution, observations_left = queue.get()
        current_iteration = len(current_solution)
        logger.debug("processing path from %s (iter=%2d, cost=%2d)", current_state, current_iteration, current_cost)

        if current_state in best_iteration_cost[current_iteration] \
                and best_iteration_cost[current_iteration][current_state] <= current_cost:
            logger.debug("skipping current path (cost=%2d), better path exists (cost=%2d)",
                         current_cost, best_iteration_cost[current_iteration][current_state])
            continue

        else:
            best_iteration_cost[current_iteration][current_state] = current_cost

        if observations_left == "":
            logger.debug("solution found, saving result %s (cost=%2d)", current_solution, current_cost)

            finished_paths.append((current_cost, current_solution))
            continue

        current_observation = observations_left[-observation_offset:]
        for possible_bit_value in [0, 1]:
            transition_observation = state_to_str(emissions[current_state][possible_bit_value])
            transition_cost = hamming_distance(transition_observation, current_observation)

            next_state = transitions[current_state][possible_bit_value]
            queue.put((
                transition_cost + current_cost,
                next_state,
                current_solution.copy() + [possible_bit_value],
                observations_left[:-observation_offset])
            )

    return finished_paths
