import pytest
from . import utils


@pytest.fixture(params=["bm"], ids=["bm"])
def model_setup(request):
    return utils.model_setup(request)


def test_particle_filter_rb_for(model_setup):
    utils.test_particle_filter_rb_for(**model_setup)


def test_particle_filter_rb_history(model_setup):
    utils.test_particle_filter_rb_history(**model_setup)


def test_particle_filter_rb_deriv(model_setup):
    utils.test_particle_filter_rb_deriv(**model_setup)
