from enum import Enum


class FineTuneMode(Enum):
    HEAD = 'head'
    BACKBONE = 'backbone'
    FULL = 'full'
