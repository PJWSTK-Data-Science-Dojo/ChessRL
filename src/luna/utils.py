"""
    Util Classes/functions for luna
"""
import os

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        super(AverageMeter, self).__init__()

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def find_project_root(marker_file="main.py"):
    """Finds the project root directory by searching for a marker file."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    # Go up directories until the marker file is found or we hit the filesystem root
    while True:
        if os.path.exists(os.path.join(current_dir, marker_file)):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir: # Reached filesystem root
            raise FileNotFoundError(f"Project root marker '{marker_file}' not found.")
        current_dir = parent_dir

def get_absolute_path(relative_path: str, base_dir: str):
    """Gets the absolute path by joining base_dir and relative_path."""
    # Normalize the path to handle '..' etc.
    return os.path.abspath(os.path.join(base_dir, relative_path))