from .MES import MES
from .PES import PES
from .MES import MES_test


acquisition_functions = {
    "PES": PES,
    "MES": MES,
    "MES_test": MES_test
}
