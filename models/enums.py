from enum import Enum

class Currency(str, Enum):
    USD = "USD"
    KHR = "KHR"

class Language(str, Enum):
    EN = "EN"
    CH = "CH"