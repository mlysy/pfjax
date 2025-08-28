import pytest
from . import utils

model_names = ["lv_model", "bm_model", "pg_model"]
model_ids = ["lv", "bm", "pg"]

@pytest.fixture(params=model_names[0:1], ids=model_ids[0:1])
def model_setup(request):
    if request.param == "lv_model":
        return utils.lv_setup()
    if request.param == "bm_model":
        return utils.bm_setup()
    if request.param == "pg_model":
        return utils.pg_setup()
    raise ValueError(f"Unknown model type: {request.param}")

def test_particle_filter_rb_for(model_setup):
    utils.test_particle_filter_rb_for(**model_setup)

def test_particle_filter_rb_history(model_setup):
    utils.test_particle_filter_rb_history(**model_setup)

def test_particle_filter_rb_deriv(model_setup):
    utils.test_particle_filter_rb_deriv(**model_setup)

