import logging
import typing
from collections import deque

logger = logging.getLogger("filter")


class ConvolutionFilter(object):

    def __init__(self, stage_count: int, feedback_masks: typing.List[int]):
        self.stage_count = stage_count + 1  # (+1) because of one stage for current bit
        self.feedback_masks = feedback_masks

        self._filter = deque()

    def __str__(self):
        return f"[MSB] {' '.join(map(lambda x: str(x), self._filter))} [LSB]"

    @property
    def output(self):
        outputs = [0 for _ in self.feedback_masks]

        # for each bit in convolution filter do
        # (filter bits are in MSB->LSB order)
        for filter_bit, filter_bit_index in zip(self._filter, range(len(self._filter) - 1, -1, -1)):
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
