import os


def normalize_path(path):
    return os.path.abspath(os.path.expanduser(path))

