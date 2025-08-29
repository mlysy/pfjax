import pytest
from . import utils


@pytest.fixture(params=["lv", "bm", "pg"], ids=["lv", "bm", "pg"])
def model_setup(request):
    return utils.model_setup(request)


@pytest.fixture(params=["lv", "pg"], ids=["lv", "pg"])
def two_model_setup(request):
    return utils.model_setup(request)


def test_loglik_full_for(model_setup):
    utils.test_loglik_full_for(**model_setup)


def test_loglik_full_jit(model_setup):
    utils.test_loglik_full_jit(**model_setup)


def test_loglik_full_models(two_model_setup):
    utils.test_loglik_full_models(**two_model_setup)
