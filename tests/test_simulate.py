import pytest
from . import utils


@pytest.fixture(params=["lv", "bm", "pg", "ss"], ids=["lv", "bm", "pg", "ss"])
def model_setup(request):
    return utils.model_setup(request)


@pytest.fixture(params=["lv", "pg"], ids=["lv", "pg"])
def two_model_setup(request):
    return utils.model_setup(request)


def test_simulate_jit(model_setup):
    utils.test_simulate_jit(**model_setup)


def test_simulate_for(model_setup):
    utils.test_simulate_for(**model_setup)


def test_simulate_models(two_model_setup):
    utils.test_simulate_models(**two_model_setup)
