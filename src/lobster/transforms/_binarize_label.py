import torch


class BinarizeLabelTransform:
    """Transform that converts continuous labels to binary classes based on a threshold.

    This is useful for converting regression labels to classification labels on-the-fly
    without modifying the original data.

    Parameters
    ----------
    threshold : float
        Threshold value for binarization. Values > threshold become class 1,
        values <= threshold become class 0.

    Examples
    --------
    >>> transform = BinarizeLabelTransform(threshold=1.0)
    >>> label = 1.5  # Continuous value
    >>> binary_label = transform(label)  # Returns tensor([1])
    >>>
    >>> label = 0.5
    >>> binary_label = transform(label)  # Returns tensor([0])
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, label: float | int | torch.Tensor) -> torch.Tensor:
        """Convert continuous label to binary class.

        Parameters
        ----------
        label : float | int | torch.Tensor
            Continuous label value

        Returns
        -------
        torch.Tensor
            Binary class label (0 or 1) as a tensor
        """
        if isinstance(label, torch.Tensor):
            return (label > self.threshold).long()
        return torch.tensor(1 if label > self.threshold else 0, dtype=torch.long)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"
