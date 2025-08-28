import pytest
from . import utils

@pytest.fixture(params=["lv_model", "bm_model", "pg_model"], ids=["lv", "bm", "pg"])
def model_setup(request):
    if request.param == "lv_model":
        return utils.lv_setup()
    elif request.param == "bm_model":
        return utils.bm_setup()
    elif request.param == "pg_model":
        return utils.pg_setup()
    else:
        raise ValueError(f"Unknown model type: {request.param}")

@pytest.fixture(params=["lv", "pg"], ids=["lv", "pg"])
def model_pair(request):
    if request.param == "lv":
        obj = utils.lv_setup()
        return (obj['model'](**obj['model_args']), obj['model2'](**obj['model_args']), obj)
    elif request.param == "pg":
        obj = utils.pg_setup()
        return (obj['model'](**obj['model_args']), obj['model2'](**obj['model_args']), obj)
    else:
        raise ValueError(f"Unknown model type: {request.param}")
    
def test_loglik_full_for(model_setup):
    utils.test_loglik_full_for(**model_setup)

def test_loglik_full_jit(model_setup):
    utils.test_loglik_full_jit(**model_setup)

def test_loglik_full_models(model_pair):
    model, model2, obj = model_pair
    key = obj['key']
    n_obs = obj['n_obs']
    x_init = obj['x_init']
    theta = obj['theta']
    model_args=obj["model_args"]
    utils.test_loglik_full_models(
        model1=model, 
        model2=model2, 
        key=key, 
        n_obs=n_obs, 
        x_init=x_init, 
        theta=theta, 
        model_args=model_args
        )