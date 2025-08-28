import pytest
from . import utils

@pytest.fixture(params=["ot_model"], ids=["ot"])
def model_setup(request):
    if request.param == "ot_model":
        return utils.ot_setup()
    raise ValueError(f"Unknown model type: {request.param}")

def test_resample_ot_sinkhorn(model_setup):
    utils.test_resample_ot_sinkhorn(**model_setup)

def test_resample_ot_jit(model_setup):
    utils.test_resample_ot_jit(**model_setup)
