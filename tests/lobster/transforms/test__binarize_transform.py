from lobster.transforms import BinarizeTransform


def test_binarize_transform():
    transform = BinarizeTransform(0.5)
    assert transform(0.6) == 1
    assert transform(0.4) == 0
