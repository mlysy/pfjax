import pytest
from . import utils

@pytest.fixture(params=["lv_model", "bm_model", "pg_model"], ids=["lv", "bm", "pg"])
def model_setup(request):
    if request.param == "lv_model":
        return utils.lv_setup()
    if request.param == "bm_model":
        return utils.bm_setup()
    if request.param == "pg_model":
        return utils.pg_setup()
    raise ValueError(f"Unknown model type: {request.param}")

def test_particle_filter_for(model_setup):
    utils.test_particle_filter_for(**model_setup)

def test_particle_filter_deriv(model_setup):
    utils.test_particle_filter_deriv(**model_setup)
