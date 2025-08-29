import pytest
from . import utils


@pytest.fixture(params=["ot"], ids=["ot"])
def ot_setup(request):
    if request.param == "ot":
        return utils.ot_setup()
    raise ValueError(f"Unknown model type: {request.param}")


def test_resample_ot_sinkhorn(ot_setup):
    utils.test_resample_ot_sinkhorn(**ot_setup)


def test_resample_ot_jit(ot_setup):
    utils.test_resample_ot_jit(**ot_setup)
