import sys
import os

if sys.version_info.major == 2:
    import imp
else:
    import importlib

__all__ = ['import_from_directory']


def import_from_directory(library, path):
    """Import a library from a directory outside the path.

    Parameters
    ----------
    library: string
        The name of the library.
    path: string
        The path of the folder containing the library.
    """
    module_path = os.path.abspath(path)
    version = sys.version_info

    if version.major == 2:
        f, filename, desc = imp.find_module(library, [module_path])
        return imp.load_module(library, f, filename, desc)
    else:
        sys.path.append(module_path)
        return importlib.import_module(library)
