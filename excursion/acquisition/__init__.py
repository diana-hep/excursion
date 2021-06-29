from .MES import MES
from .PES import PES
from .MES import MES_test
from .batch import batchGrid

acquisition_functions = {
    "PES": PES,
    "MES": MES,
    "MES_test": MES_test
}
