import logging
from collections import deque

logger = logging.getLogger("filter")


class ConvolutionFilter(object):

    def __init__(self, stage_count: int, msb_feedback: int, lsb_feedback: int):
        self.stage_count = stage_count + 1  # (+1) because of one stage for current bit
        self.msb_feedback = msb_feedback
        self.lsb_feedback = lsb_feedback

        self._filter = deque()

    def __str__(self):
        return f"[MSB] {' '.join(map(lambda x: str(x), self._filter))} [LSB]"

    @property
    def output(self):
        output_msb = 0
        output_lsb = 0

        # for each bit in convolution filter do
        # (filter bits are in MSB->LSB order)
        for filter_bit, filter_bit_index in zip(self._filter, range(len(self._filter) - 1, -1, -1)):
            # calculate bit mask from current index
            filter_bit_value = pow(2, filter_bit_index)

            # does bit mask match with allowed MSB feedback value?
            if filter_bit_value & self.msb_feedback:
                output_msb ^= filter_bit  # perform binary XOR

            # does bit mask match with allowed LSB feedback value?
            if filter_bit_value & self.lsb_feedback:
                output_lsb ^= filter_bit  # perform binary XOR

            logger.debug("bit_mask=%3d bit_value=%d out=%d%d", filter_bit_value, filter_bit, output_msb, output_lsb)

        # output two resulting bits
        return [output_msb, output_lsb]

    @property
    def empty(self):
        return not bool(self._filter)

    def initialize(self):
        self._filter = deque([0 for _ in range(self.stage_count)])
        logger.debug("%s - convolution filter initialized", self)

    def insert_and_shift(self, element: int):
        # insert new element at the MSB position - "shift" everything right
        self._filter.appendleft(element)
        # remove last extra element (at the LSB position)
        self._filter.pop()
        logger.debug("%s - inserted '%d' and shifted right", self, element)

    def shift(self):
        # "shift" everything right
        self._filter.pop()
        logger.debug("%s - shifted right", self)
