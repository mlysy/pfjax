import pytest
from . import utils

@pytest.fixture(params=["lv_model", "pg_model"], ids=["lv", "pg"])
def model_setup(request):
    if request.param == "lv_model":
        return utils.lv_setup()
    if request.param == "pg_model":
        return utils.pg_setup()
    raise ValueError(f"Unknown model type: {request.param}")

@pytest.fixture(params=["lv"], ids=["lv"])
def model_pair(request):
    if request.param == "lv":
        obj = utils.lv_setup()
        return (obj['model'](**obj['model_args']), obj['model2'](**obj['model_args']), obj)
    else:
        raise ValueError(f"Unknown model type: {request.param}")

def test_sde_state_sample_for(model_pair):
    model, model2, obj = model_pair
    key = obj['key']
    n_obs = obj['n_obs']
    x_init = obj['x_init']
    theta = obj['theta']
    n_particles = obj['n_particles']
    model_args = obj['model_args']
    n_obs = obj['n_obs']
    utils.test_sde_state_sample_for(model=model, model2=model2, key=key, x_init=x_init, 
                                  theta=theta, n_particles=n_particles, 
                                  model_args=model_args, n_obs=n_obs)

def test_sde_state_lpdf_for(model_pair):
    model, model2, obj = model_pair
    key = obj['key']
    n_obs = obj['n_obs']
    x_init = obj['x_init']
    theta = obj['theta']
    n_particles = obj['n_particles']
    model_args = obj['model_args']
    n_obs = obj['n_obs']
    utils.test_sde_state_lpdf_for(model=model, model2=model2, key=key, x_init=x_init, 
                                  theta=theta, n_particles=n_particles, 
                                  model_args=model_args, n_obs=n_obs)

def test_bridge_step_for(model_setup):
    utils.test_bridge_step_for(**model_setup)
