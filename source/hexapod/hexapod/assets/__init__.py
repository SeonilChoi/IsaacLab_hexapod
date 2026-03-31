import os

ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.dirname(__file__))

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")

from .robots import *