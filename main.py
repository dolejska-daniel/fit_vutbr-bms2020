import logging
import sys
from argparse import ArgumentParser
from enum import Enum, auto


class OperationMode(Enum):
    ENCODE = auto()
    DECODE = auto()


def run(args):
    logging.info("running in %s", args["mode"])
    if args["mode"] is OperationMode.ENCODE:
        # encoding mode - ASCII characters on STDIN, binary sequence to STDOUT
        from convolutional_encoder import ConvolutionalEncoder
        encoder = ConvolutionalEncoder(memory_stage_count, feedback_masks, not args["no_encoder_filter"])

        def print_encoded(d, inline: bool = False):
            print("".join(map(lambda x: "".join(map(lambda y: str(y), x)), d)), end="" if inline else "\n")

        if args["stream"]:
            while data_in := sys.stdin.read(1):
                logging.info("encoding input data: %s", repr(data_in))
                data_out = encoder.encode(data_in, flush_filter=False)

                # print out resulting data
                logging.info("encoded as: %s", data_out)
                print_encoded(data_out, inline=True)

            logging.info("flushing filter contents")
            data_out = encoder.encode("", flush_filter=True)

        else:
            data_in = "".join(sys.stdin.readlines())
            logging.info("encoding input data: %s", repr(data_in))
            data_out = encoder.encode(data_in)

        # print out resulting data
        logging.info("encoded as: %s", data_out)
        print_encoded(data_out)

    elif args["mode"] is OperationMode.DECODE:
        # decoding mode - binary sequence on STDIN, ASCII characters to STDOUT
        from convolutional_decoder import ConvolutionalDecoder
        decoder = ConvolutionalDecoder(memory_stage_count, feedback_masks)

        def print_best_decoded(d, inline: bool = False):
            print(d[0][1], end="" if inline else "\n")

        if args["stream"]:
            while data_in := sys.stdin.read(8):
                logging.info("decoding input data: %s", repr(data_in))
                data_out = decoder.decode(data_in)

                # print out resulting data
                logging.info("most probable encoded results (cost, data): %s", data_out)
                print_best_decoded(data_out, inline=True)

            print()

        else:
            data_in = sys.stdin.readline().strip()

            logging.info("decoding input data: %s", repr(data_in))
            data_out = decoder.decode(data_in)

            # print out resulting data
            logging.info("most probable encoded results (cost, data): %s", data_out)
            print_best_decoded(data_out)

    else:
        raise RuntimeError("Unknown operation mode selected.")


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
    params.add_argument("--no-encoder-filter", action="store_true",
                        help="encoder will accept any ASCII characters as input", )

    arguments = parser.parse_args().__dict__
    if not arguments["mode"]:
        parser.error("mode option (either -e or -d) must be present")

    if len(arguments["params"]) < 2:
        parser.error("invalid parameter specification")

    memory_stage_count = arguments["params"][0]
    feedback_masks = arguments["params"][1:]

    log_level = logging.ERROR - min(arguments["verbose"], logging.ERROR // 10) * 10
    logging.basicConfig(
        format="%(asctime)s %(levelname)s (%(name)s): %(message)s",
        level=log_level,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    try:
        run(arguments)

    except:
        logging.exception("program encountered an error while running")
