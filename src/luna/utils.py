"""
Util Classes/functions for luna
"""


class AverageMeter:
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def __repr__(self) -> str:
        return f"{self.avg:.2e}"

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
