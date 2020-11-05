import logging
import typing
import re
from collections import deque

from convolution_filter import ConvolutionFilter

logger = logging.getLogger("encoder")


class ConvolutionalEncoder(object):

    def __init__(self, stage_count: int, feedback_masks: typing.List[int], filter_input: bool = True):
        """
        Initializes convolutional encoder.

        :param stage_count: number of memory blocks
        :param feedback_masks: list of feedback masks of memory blocks
        """
        self.filter = ConvolutionFilter(stage_count, feedback_masks)
        self.filter_input = filter_input

    @classmethod
    def filter_data_in(cls, data_in: str) -> str:
        return "".join(re.findall(r"[A-z0-9]", data_in))

    @classmethod
    def str_to_int_binary(cls, data_in: str) -> typing.List[int]:
        """
        Converts ASCII string to list of integers (1, 0) representing binary
        encoded characters.

        :param data_in: ASCII string to be converted
        :return: binary sequence
        """
        data_binary = []

        # for each character in input data
        for char in data_in:
            # get ascii value of the character
            char_numeric = ord(char)
            if char_numeric > 255:
                raise RuntimeError("Unable to encode input character '{:s}' as ASCII character.".format(char))

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

        return data_binary

    def encode(self, data_in: str, flush_filter=True) -> typing.List[typing.List[int]]:
        """
        Encodes provided ASCII string to binary sequence.

        :param data_in: ASCII string to be encoded
        :param flush_filter: should convolution filter values be flushed?
        :return:
        """
        if self.filter_input:
            # remove undesired input content
            data_in = self.filter_data_in(data_in)

        # convert input to "binary" integers
        data_in_binary = self.str_to_int_binary(data_in)
        # reverse the order from MSB->LSB to LSB->MSB
        # this allows for simple `deque.popleft()` to get data bit by bit
        data_in_binary.reverse()

        # create collection to `pop` from
        data_in_binary = deque(data_in_binary)
        # create output collection to `appendleft` to
        data_out = deque([])

        if self.filter.empty:
            # initialize convolution stages
            self.filter.initialize(first_bit=data_in_binary.popleft())

        # until there is no content in input data list do
        while data_in_binary:
            # get current bit from the input data
            current_bit = data_in_binary.popleft()

            # calculate encoder output for current data bit and filter state
            data_out.appendleft(self.filter.output)

            # update convolution filter - shift to right and add current bit
            self.filter.insert_and_shift(current_bit)

        # until there is no content in convolution filter do
        while flush_filter and not self.filter.empty:
            # flush values in the filter state one by one
            data_out.appendleft(self.filter.output)

            # empty the filter bits
            self.filter.shift()

        return list(data_out)
