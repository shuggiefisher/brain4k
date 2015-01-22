#!python
import logging

from brain4k.pipeline import imagenet_parser


logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
   imagenet_parser()