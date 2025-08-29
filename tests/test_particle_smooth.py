import pytest
from . import utils


@pytest.fixture(params=["lv", "bm", "pg"], ids=["lv", "bm", "pg"])
def model_setup(request):
    return utils.model_setup(request)


def test_particle_smooth_for(model_setup):
    utils.test_particle_smooth_for(**model_setup)


def test_particle_smooth_jit(model_setup):
    utils.test_particle_smooth_jit(**model_setup)
