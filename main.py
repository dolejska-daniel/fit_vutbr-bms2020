import logging
import sys
from argparse import ArgumentParser
from enum import Enum, auto


class OperationMode(Enum):
    ENCODE = auto()
    DECODE = auto()


if __name__ == '__main__':
    parser = ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-e", action="store_const", const=OperationMode.ENCODE, dest="mode")
    mode_group.add_argument("-d", action="store_const", const=OperationMode.DECODE, dest="mode")

    parser.add_argument("--stream", action="store_true", help="run program in stream mode", )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="sets verbosity of logging (1-3)", )

    args = parser.parse_args().__dict__
    if not args["mode"]:
        parser.error("mode option (either -e or -d) must be present")

    log_level = logging.ERROR - min(args["verbose"], logging.ERROR // 10) * 10
    logging.basicConfig(
        format="%(asctime)s %(levelname)s (%(name)s): %(message)s",
        level=log_level,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    if args["mode"] is OperationMode.ENCODE:
        from convolutional_encoder import ConvolutionalEncoder
        encoder = ConvolutionalEncoder(5, [53, 46])

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
        raise NotImplemented("This mode has not been implemented yet.")

    else:
        raise RuntimeError("Unknown operation mode selected.")
