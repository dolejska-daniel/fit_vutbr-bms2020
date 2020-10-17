# Convolutional Encoder and Decoder
> v0.1

## Introduction
This project implements simple configurable convolutional encoder and decoder.

### Requirements
Only `Python >= 3.8` is required.

### Documentation
Code is well commented, everything necessary should be contained within the source codes of the project. 

## Usage
Program requires specification of operation mode `-e` for **encoding**, `-d` for **decoding**.
Encoding mode expects ASCII characters on `STDIN`.
Those characters are encoded to binary sequence, which is output to `STDOUT` after encountering `EOF` on `STDIN`.
Decoding mode expects `MSB->LSB` binary sequence on `STDIN`.
This sequence is then decoded to ASCII characters, which are then output to `STDOUT` after encoutering `EOF` on `STDIN`.
```
python3.8 main.py {-e | -d}
```

Internal encoder and decoder configuration can be explicitly specified using `--params` option.
  - `X` corresponds to a number of convolution memory blocks (default: 5)
  - `Y`, `Z` and `other values` correspond to feedback memory bit masks per each output bit (defaults: 53 46)
```
python3.8 main.py {-e | -d} [--params X Y Z ...]
```

You can use option `-h` or `--help` for more information about the program:
```
python3.8 main.py [-h | --help]
```
