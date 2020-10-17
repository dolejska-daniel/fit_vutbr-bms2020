import logging
from collections import deque

import typing

from convolution_filter import ConvolutionFilter

logger = logging.getLogger("encoder")


class ConvolutionalEncoder(object):

    def __init__(self, stage_count: int, feedback_masks: typing.List[int]):
        self.filter = ConvolutionFilter(stage_count, feedback_masks)

    @classmethod
    def _str_to_int_binary(cls, data_in: str) -> typing.List[int]:
        data_binary = []

        # for each character in input data
        for char in data_in:
            # get ascii value of the character
            char_numeric = ord(char)
            # convert to binary string and remove '0b' prefix
            char_binary_str = str(bin(char_numeric))[2:]

            # create padding to 8 bits
            char_binary = [0 for _ in range(8 - len(char_binary_str))]
            # convert string characters [01] to integers
            char_binary += [int(binary_char) for binary_char in char_binary_str]
            logger.debug("converted character '%s' (%d) to binary %s", char, char_numeric,
                         ''.join(map(lambda x: str(x), char_binary)))

            logger.debug("storing in reverse order %s", ''.join(map(lambda x: str(x), char_binary)))
            data_binary += char_binary

        # reverse the order from MSB->LSB to LSB->MSB
        # this allows for simple `deque.popleft()` to get data bit by bit
        data_binary.reverse()

        return data_binary

    def encode(self, data_in: str, flush_filter=True):
        # convert input to "binary" integers
        data_in_binary = self._str_to_int_binary(data_in)
        # create collection to `pop` from
        data_in_binary = deque(data_in_binary)
        # create output collection to `appendleft` to
        data_out = deque([])

        # initialize convolution stages
        self.filter.initialize()

        # until there is no content in input data list do
        while data_in_binary:
            # get current bit from the input data
            current_bit = data_in_binary.popleft()

            data_out.appendleft(self.filter.output_for(current_bit))

            # update convolution filter - shift to right and add current bit
            self.filter.insert_and_shift(current_bit)

        # until there is no content in convolution filter do
        while flush_filter and not self.filter.empty:
            data_out.appendleft(self.filter.output_for(None))

            # empty the filter bits
            self.filter.shift()

        return data_out
