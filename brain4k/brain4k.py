#!python
import logging

from pipeline import imagenet_parser


logging.basicConfig(level=logging.DEBUG)


def run():
    imagenet_parser()