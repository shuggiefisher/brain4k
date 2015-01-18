#!python
import logging

from pipeline import imagenet_parser


logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
   # main(sys.argv[1:])
   imagenet_parser()