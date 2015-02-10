import os

from jinja2 import Environment, FileSystemLoader

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

template_env = Environment(loader=FileSystemLoader(ROOT_PATH))
