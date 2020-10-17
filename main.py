import logging
import sys
from argparse import ArgumentParser
from collections import deque
from enum import Enum, auto


class OperationMode(Enum):
    ENCODE = auto()
    DECODE = auto()


if __name__ == '__main__':
    parser = ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-e", action="store_const", const=OperationMode.ENCODE, dest="mode",
                            help="perform encoding on input values")
    mode_group.add_argument("-d", action="store_const", const=OperationMode.DECODE, dest="mode",
                            help="perform decoding on input values")

    parser.add_argument("-v", "--verbose", action="count", default=0, help="sets verbosity of logging (1-3)", )

    params = parser.add_argument_group("algorithm parameters")
    params.add_argument("--params", nargs="+", type=int, default=[5, 53, 46], metavar="X Y Z",
                        help="customizable program parameters (X is number of memory blocks; "
                             "Y,Z and other values are feedback memory bit masks per each output bit) "
                             "[defaults: 5 53 46]")
    params.add_argument("--stream", action="store_true", help="run program in stream mode", )

    args = parser.parse_args().__dict__
    if not args["mode"]:
        parser.error("mode option (either -e or -d) must be present")

    if len(args["params"]) < 2:
        parser.error("invalid parameter specification")

    memory_stage_count = args["params"][0]
    feedback_masks = args["params"][1:]

    log_level = logging.ERROR - min(args["verbose"], logging.ERROR // 10) * 10
    logging.basicConfig(
        format="%(asctime)s %(levelname)s (%(name)s): %(message)s",
        level=log_level,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    if args["mode"] is OperationMode.ENCODE:
        from convolutional_encoder import ConvolutionalEncoder
        encoder = ConvolutionalEncoder(memory_stage_count, feedback_masks)

        if args["stream"]:
            while data_in := sys.stdin.read(1):
                data_out = encoder.encode(data_in)

                # print out resulting data
                for out_pair in data_out:
                    for out_char in out_pair:
                        print(out_char, end="", flush=True)

        else:
            data_out = encoder.encode("\n".join(sys.stdin.readlines()))

            # print out resulting data
            for out_pair in data_out:
                for out_char in out_pair:
                    print(out_char, end="", flush=True)

    elif args["mode"] is OperationMode.DECODE:
        from viterbi import viterbi_own

        paths = viterbi_own(memory_stage_count, feedback_masks)
        result_binary = paths[0][1]

        result = deque([])
        byte_value = 0
        for bit_index, bit in enumerate(result_binary):
            if bit_index % 8 == 0 and bit_index > 0:
                result.appendleft(chr(byte_value))
                byte_value = 0
                if bit_index + 8 > len(result_binary):
                    break

            byte_value += pow(2, bit_index % 8) * bit

        result = "".join(result)
        print(result)

    else:
        raise RuntimeError("Unknown operation mode selected.")
