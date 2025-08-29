import pytest
from . import utils


@pytest.fixture(params=["lv"], ids=["lv"])
def two_model_setup(request):
    return utils.model_setup(request)


def test_sde_state_sample_for(two_model_setup):
    utils.test_sde_state_sample_for(**two_model_setup)


def test_sde_state_lpdf_for(two_model_setup):
    utils.test_sde_state_lpdf_for(**two_model_setup)


# def test_bridge_step_for(model_setup):
#     utils.test_bridge_step_for(**model_setup)
