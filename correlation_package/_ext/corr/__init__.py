from torch.utils.cpp_extension import load
from correlation_package._ext.corr import lib as _lib  

__all__ = []

def _import_symbols(locals_):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        locals_[symbol] = fn 
        __all__.append(symbol)

_import_symbols(locals())