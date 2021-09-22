import sys
import contextlib

@contextlib.contextmanager
def open_file_or_stdout(filename=None):
    if filename is None or filename == '-':
        handle = sys.stdout
    else:
        handle = open(filename, 'w')
    try:
        yield handle
    finally:
        if handle is not sys.stdout:
            handle.close()
