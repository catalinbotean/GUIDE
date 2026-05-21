import guide


def test_version():
    assert guide.__version__ == "0.0.1"


def test_subpackages_importable():
    from guide import data, eval, models, training, utils  # noqa: F401
