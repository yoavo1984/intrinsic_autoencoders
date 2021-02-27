from enum import Enum


class DataType(str, Enum):
    LINEAR = 'linear'
    POWER_2 = 'power_2'
    DEPENDENT = 'dependent'
