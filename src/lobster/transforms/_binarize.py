from typing import List

from ._lambda import Lambda


class BinarizeTransform(Lambda):
    def __init__(self, threshold: float):
        """
        Parameters
        ----------
        threshold : float
            The threshold value for binarization.

        """
        super().__init__(lambda x: 1 if x > threshold else 0, float)

    def _check_inputs(self, x: List[float]):
        if not isinstance(x[0], float):
            raise TypeError(f"Expected input to be of type float, got {type(x[0])}")
