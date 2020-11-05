import logging
import typing
from collections import deque

from utils import State

logger = logging.getLogger("filter")


class ConvolutionFilter(object):

    def __init__(self, stage_count: int, feedback_masks: typing.List[int]):
        """
        Initializes convolution filter.

        :param stage_count: number of memory blocks
        :param feedback_masks: list of feedback masks of memory blocks
        """
        self.stage_count = stage_count
        self.feedback_masks = feedback_masks

        self._memory = deque()

    def __str__(self):
        return str(self.state)

    @property
    def state(self) -> State:
        """ Returns current filter state. """
        return State(list(self._memory))

    @property
    def empty(self) -> bool:
        """ Returns true is the filter is empty. """
        return not bool(self._memory)

    def initialize(self, state: State = None, first_bit: int = 0):
        """
        Initializes filter state either to provided state or default state 0s.

        :param state: target filter state
        """
        if state and len(state) != self.stage_count:
            logger.critical("invalid initialization state vector %s", state)
            raise RuntimeError("Tried to initialize invalid filter state.")

        self._memory = state if state else deque([first_bit] + [0 for _ in range(self.stage_count - 1)])
        logger.debug("%s - convolution filter initialized", self)

    def insert_and_shift(self, memory_bit: int):
        """
        Shifts the filter state right and inserts new element at the MSB position.

        :param memory_bit: bit value (0, 1) to be stored
        """
        # insert new element at the MSB position - "shift" everything right
        self._memory.appendleft(memory_bit)
        # remove last extra element (at the LSB position)
        self._memory.pop()
        logger.debug("%s - inserted '%d' and shifted right", self, memory_bit)

    def shift(self):
        """ Shifts the filter state right (does not add any new elements - can empty the filter). """
        # "shift" everything right
        self._memory.pop()
        logger.debug("%s - shifted right", self)
    
    @property
    def output(self) -> typing.List[int]:
        """ Calculates result for current filter state. """
        outputs = [0 for _ in self.feedback_masks]
        logger.debug("%s - calculating output with current state", self.state)

        # for each bit in convolution filter do
        # (filter bits are in MSB->LSB order)
        for filter_bit, filter_bit_index in zip(self.state, range(len(self.state) - 1, -1, -1)):
            # calculate bit mask from current index
            filter_bit_value = pow(2, filter_bit_index)

            # for each feedback mask (corresponding to result bit)
            for output_bit_index, feedback_mask in enumerate(self.feedback_masks):
                # does bit mask match with feedback mask?
                if filter_bit_value & feedback_mask:
                    outputs[output_bit_index] ^= filter_bit  # perform binary XOR

            logger.debug("bit_mask=%3d bit_value=%d out=%s", filter_bit_value, filter_bit, outputs)

        # output two resulting bits
        return outputs
