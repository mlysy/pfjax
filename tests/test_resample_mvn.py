import pytest
from . import utils


@pytest.fixture(params=["bm"], ids=["bm"])
def model_setup(request):
    return utils.model_setup(request)


def test_resample_mvn_for(model_setup):
    utils.test_resample_mvn_for(**model_setup)


def test_resample_mvn_shape(model_setup):
    utils.test_resample_mvn_shape(**model_setup)


def test_resample_mvn_jit(model_setup):
    utils.test_resample_mvn_jit(**model_setup)
