import pytest
from . import utils

@pytest.fixture(params=["bm_model"], ids=["bm"])
def model_setup(request):
    if request.param == "bm_model":
        return utils.bm_setup()
    raise ValueError(f"Unknown model type: {request.param}")

def test_resample_mvn_for(model_setup):
    utils.test_resample_mvn_for(**model_setup)

def test_resample_mvn_shape(model_setup):
    utils.test_resample_mvn_shape(**model_setup)

def test_resample_mvn_jit(model_setup):
    utils.test_resample_mvn_jit(**model_setup)
