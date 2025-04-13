from typing import Dict

import abc


class TransformLambda:
    """
    Lambda takes a dictionary, which represents a single data point from the dataset
    and appends to this dictionary. It should change the already existing content of the
    dictionary (not add new ones).
    """

    @abc.abstractmethod
    def __call__(self, sample: Dict) -> Dict:
        pass
