import itertools
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
import jax.tree_util as jtu
import ott
import pandas as pd
import pfjax as pf
import pfjax.experimental.models as models_exp
import pfjax.mcmc as mcmc
import pfjax.models as models
import pfjax.particle_resamplers as resamplers
import pfjax.test.models as models_test
import pfjax.test.utils as test
import pfjax.utils as utils
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# --- general-purpose utilities ------------------------------------------------


def rel_err(X1, X2):
    """
    Relative error between two JAX arrays.

    Adds 0.1 to the denominator to avoid nan's when its equal to zero.
    """
    x1 = jnp.atleast_1d(X1).ravel() * 1.0
    x2 = jnp.atleast_1d(X2).ravel() * 1.0
    return jnp.max(jnp.abs(x1 - x2) / (0.1 + jnp.abs(x1)))


def assert_equal(x1, x2, tol=1e-6, context=""):
    """
    Relative error check for arrays or dictionaries.

    Parameters
    ----------
    x1, x2 : array-like or dict
        Objects to compare.
    tol : float, optional
        Relative error tolerance. Default is 1e-6.
    context : str, optional
        Extra string to include in assertion messages
        (e.g. "failed at index i, key k, or case c").
    """
    if isinstance(x1, dict):
        for k in x1.keys():
            assert rel_err(x1[k], x2[k]) < tol, f"failed at key '{k}'" + (
                f", {context}" if context else ""
            )
    else:
        assert rel_err(x1, x2) < tol, f"failed" + (f" at {context}" if context else "")


def var_sim(key, size):
    """
    Generate a variance matrix of given size.
    """
    Z = random.normal(key, (size, size))
    return jnp.matmul(Z.T, Z)


# def expand_grid(**kwargs):
#     """
#     JAX equivalent of expand_grid in R.

#     Unlike R, leftmost vectors are changing fastest.
#     """
#     keys = list(kwargs)
#     out = jnp.meshgrid(*[kwargs[k] for k in keys])
#     return {keys[i]: jnp.ravel(out[i]) for i in jnp.arange(len(out))}


def expand_grid(**kwargs):
    """
    Create a dataframe from every combination of given values.
    """
    rows = itertools.product(*kwargs.values())
    return pd.DataFrame.from_records(rows, columns=kwargs.keys())


# --- setup methods ------------------------------------------------------------


def model_setup(request):
    """
    Model selector to be invoked by `pytest.fixture()`.
    """
    if request.param == "lv":
        return lv_setup()
    elif request.param == "bm":
        return bm_setup()
    elif request.param == "pg":
        return pg_setup()
    else:
        raise ValueError(f"Unknown model type: {request.param}")


def bm_setup():
    """
    Setup function for BMModel tests.

    This initializes key variables required for running Brownian Motion (BMModel)
    tests, including model parameters, number of observations, initial state,
    number of particles, and the model class reference.

    Returns
    -------
    None
        Sets the following attributes on self:
        - key : jax.random.PRNGKey
            JAX random key.
        - theta : jax.numpy.ndarray
            Model parameters for BMModel.
        - model_args : dict
            Arguments to initialize the model.
        - n_obs : int
            Number of observations.
        - x_init : jax.numpy.ndarray
            Initial state of the model.
        - n_particles : int
            Number of particles for the particle filter.
        - Model : type
            BMModel class from models.
    """

    key = random.PRNGKey(0)
    # parameter values
    mu = 5
    sigma = 1
    tau = 0.1
    theta = jnp.array([mu, sigma, tau])
    # data specification
    model_args = {"dt": 0.1}
    n_obs = 5
    x_init = jnp.array(0.0)
    # particle filter specification
    n_particles = 3
    # model specification
    Model = models.BMModel
    return {
        "key": key,
        "theta": theta,
        "model_args": model_args,
        "n_obs": n_obs,
        "x_init": x_init,
        "n_particles": n_particles,
        "model": Model,
    }


def lv_setup():
    """
    Setup function for LotVolModel tests.

    This initializes key variables required for running LotVolModel tests,
    including model parameters, number of observations, initial state,
    number of particles, and model class references.

    Returns
    -------
    None
        Sets the following attributes on self:
        - key : jax.random.PRNGKey
            JAX random key.
        - theta : jax.numpy.ndarray
            Model parameters for LotVolModel.
        - model_args : dict
            Arguments to initialize the model (e.g., dt, n_res).
        - n_obs : int
            Number of observations.
        - x_init : jax.numpy.ndarray
            Initial state of the model.
        - n_particles : int
            Number of particles for the particle filter.
        - Model : type
            LotVolModel class from models.
        - Model2 : type
            LotVolModel class from lv ().
    """

    key = random.PRNGKey(0)
    # parameter values
    alpha = 1.02
    beta = 1.02
    gamma = 4.0
    delta = 1.04
    sigma_H = 0.1
    sigma_L = 0.2
    tau_H = 0.25
    tau_L = 0.35
    theta = jnp.array([alpha, beta, gamma, delta, sigma_H, sigma_L, tau_H, tau_L])
    # data specification
    dt = 0.09
    n_res = 3
    model_args = {"dt": dt, "n_res": n_res}
    n_obs = 7
    x_init = jnp.block([[jnp.zeros((n_res - 1, 2))], [jnp.log(jnp.array([5.0, 3.0]))]])
    n_particles = 25
    Model = models_exp.LotVolModel
    Model2 = models_test.LotVolModel
    return {
        "key": key,
        "theta": theta,
        "model_args": model_args,
        "n_obs": n_obs,
        "x_init": x_init,
        "n_particles": n_particles,
        "model": Model,
        "model2": Model2,
    }


def pg_setup():
    """
    Setup function for PGNETModel tests.

    This initializes key variables required for running PGNETModel tests,
    including model parameters, number of observations, initial state,
    number of particles, and model class references.

    Returns
    -------
    None
        Sets the following attributes on self:
        - key : jax.random.PRNGKey
            JAX random key.
        - theta : jax.numpy.ndarray
            Model parameters for PGNETModel (concatenated theta and tau).
        - model_args : dict
            Arguments to initialize the model.
        - n_obs : int
            Number of observations.
        - x_init : jax.numpy.ndarray
            Initial state of the model.
        - n_particles : int
            Number of particles for the particle filter.
        - Model : type
            PGNETModel class from models.
        - Model2 : type
            PGNETModel class from models.
    """

    key = random.PRNGKey(0)
    # parameter values
    theta = jnp.array([0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1])
    tau = jnp.array([0.15, 0.2, 0.25, 0.3])
    theta = jnp.append(theta, tau)
    # data specification
    dt = 0.09
    n_res = 4
    model_args = {"dt": dt, "n_res": n_res}
    n_obs = 9
    x_init = jnp.block(
        [[jnp.zeros((n_res - 1, 4))], [jnp.log(jnp.array([8.0, 8.0, 8.0, 5.0]))]]
    )
    n_particles = 2
    Model = models.PGNETModel
    Model2 = models.PGNETModel
    return {
        "key": key,
        "theta": theta,
        "model_args": model_args,
        "n_obs": n_obs,
        "x_init": x_init,
        "n_particles": n_particles,
        "model": Model,
        "model2": Model2,
    }


# move into individual test files


def fact_setup(self):
    """
    Setup function for factorization tests.

    This initializes variables required for matrix and vector factorization
    tests, including latent and observed dimensions, random parameters,
    and joint distribution values.

    Returns
    -------
    None
        Sets the following attributes on self:
        - n_lat : int
            Number of latent dimensions (W and X).
        - n_obs : int
            Number of observed dimensions (Y).
        - mu_W : jax.numpy.ndarray
            Mean vector for W.
        - Sigma_W : jax.numpy.ndarray
            Covariance matrix for W.
        - W : jax.numpy.ndarray
            Random sample for W.
        - mu_XW : jax.numpy.ndarray
            Mean vector for X|W.
        - Sigma_XW : jax.numpy.ndarray
            Covariance matrix for X|W.
        - X : jax.numpy.ndarray
            Random sample for X.
        - A : jax.numpy.ndarray
            Observation matrix.
        - Omega : jax.numpy.ndarray
            Covariance matrix for Y.
        - Y : jax.numpy.ndarray
            Random sample for Y.
        - mu_Y : jax.numpy.ndarray
            Mean vector for Y (joint distribution).
        - Sigma_Y : jax.numpy.ndarray
            Covariance for Y (joint distribution).
        - mu : jax.numpy.ndarray
            Full joint mean vector.
        - Sigma : jax.numpy.ndarray
            Full joint covariance matrix.
    """

    key = random.PRNGKey(0)
    n_lat = 3  # number of dimensions of W and X
    n_obs = 2  # number of dimensions of Y

    # generate random values of the matrices and vectors

    key, *subkeys = random.split(key, num=4)
    mu_W = random.normal(subkeys[0], (self.n_lat,))
    Sigma_W = var_sim(subkeys[1], self.n_lat)
    W = random.normal(subkeys[2], (self.n_lat,))

    key, *subkeys = random.split(key, num=4)
    mu_XW = random.normal(subkeys[0], (self.n_lat,))
    Sigma_XW = var_sim(subkeys[1], self.n_lat)
    X = random.normal(subkeys[2], (self.n_lat,))

    key, *subkeys = random.split(key, num=4)
    A = random.normal(subkeys[0], (self.n_obs, self.n_lat))
    Omega = var_sim(subkeys[1], self.n_obs)
    Y = random.normal(subkeys[2], (self.n_obs,))

    # joint distribution using single mvn
    mu_Y = jnp.matmul(self.A, self.mu_W + self.mu_XW)
    self.Sigma_Y = (
        jnp.linalg.multi_dot([self.A, self.Sigma_W + self.Sigma_XW, self.A.T])
        + self.Omega
    )
    AS_W = jnp.matmul(self.A, self.Sigma_W)
    AS_XW = jnp.matmul(self.A, self.Sigma_W + self.Sigma_XW)
    self.mu = jnp.block([self.mu_W, self.mu_W + self.mu_XW, self.mu_Y])
    self.Sigma = jnp.block(
        [
            [self.Sigma_W, self.Sigma_W, AS_W.T],
            [self.Sigma_W, self.Sigma_W + self.Sigma_XW, AS_XW.T],
            [AS_W, AS_XW, self.Sigma_Y],
        ]
    )


def ot_setup():
    """
    Setup function for optimal transport tests.

    Initializes variables required for running optimal transport tests,
    including the PRNG key, number of particles, and problem dimension.

    Returns
    -------
    None
        Sets the following attributes on self:
        - key : PRNG key
            JAX random key.
        - n_particles : int
            Number of particles for the optimal transport test.
        - n_dim : int
            Dimension of each particle.
    """

    key = random.PRNGKey(0)
    n_particles = 12
    n_dim = 5

    return {
        "key": key,
        "n_particles": n_particles,
        "n_dim": n_dim,
    }
    # # parameter values
    # alpha = 1.02
    # beta = 1.02
    # gamma = 4.
    # delta = 1.04
    # sigma_H = .1
    # sigma_L = .2
    # tau_H = .25
    # tau_L = .35
    # theta = jnp.array([alpha, beta, gamma, delta,
    #                   sigma_H, sigma_L, tau_H, tau_L])
    # # data specification
    # dt = .09
    # n_res = 5
    # n_dim = 2
    # n_obs = 8
    # x_init = jnp.block([[jnp.zeros((n_res-1, n_dim))],
    #                   [jnp.log(jnp.array([5., 3.]))]])
    # model_args = {"dt": dt, "n_res": n_res}
    # n_particles = 12
    # Model = models.LotVolModel


# --- simulate test functions --------------------------------------------------


def test_simulate_for(model, key, n_obs, x_init, theta, model_args, **kwargs):
    """
    Test function to compare simulation implementations.

    Compares the outputs of two simulation approaches (for-loop vs scan/vmap)
    for the specified model. Checks that both the simulated measurements and
    latent states match to within numerical precision.

     Parameters
    ----------
    model : callable
        The model class or instance to use for simulation.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    model_args : dict
        Arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate the model
    model = model(**model_args)
    # simulate with for-loop
    y_meas1, x_state1 = test.simulate_for(model, key, n_obs, x_init, theta)
    # simulate without for-loop
    y_meas2, x_state2 = pf.simulate(model, key, n_obs, x_init, theta)
    assert_equal(y_meas1, y_meas2)
    assert_equal(x_state1, x_state2)


def test_simulate_jit(model, key, n_obs, x_init, theta, model_args, **kwargs):
    """
    Test function to compare jitted and non-jitted simulation and gradient calculations.

    Checks that running the simulation and its gradient with and without JAX jit produces identical results.

    Parameters
    ----------
    model : callable
        The model class or instance to use for simulation.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    model_args : dict
        Arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate the model
    model = model(**model_args)

    def obj_fun(model, key, n_obs, x_init, theta):
        """
        Objective function for gradient calculation.
        Sums all outputs from pf.simulate for use in gradient checking.

        Parameters
        ----------
        model : object
            The model instance to simulate.
        key : jax.random.PRNG Key
            JAX random key.
        n_obs : int
            Number of observations to simulate.
        x_init : ndarray
            Initial state vector for the simulation.
        theta : ndarray
            Model parameters for simulation.

        Returns
        -------
        total : float
            Sum of all elements in the simulation outputs as a scalar.
        """
        out = pf.simulate(model=model, key=key, n_obs=n_obs, x_init=x_init, theta=theta)
        return jtu.tree_reduce(lambda x, y: x + jnp.sum(y), out, jnp.array(0.0))

    # simulate without jit
    y_meas1, x_state1 = pf.simulate(model, key, n_obs, x_init, theta)
    # simulate with jit
    simulate_jit = jax.jit(pf.simulate, static_argnums=(0, 2))
    y_meas2, x_state2 = simulate_jit(model, key, n_obs, x_init, theta)

    assert_equal(y_meas1, y_meas2)
    assert_equal(x_state1, x_state2)

    # grad without jit
    grad1 = jax.grad(obj_fun, argnums=4)(model, key, n_obs, x_init, theta)
    # grad with jit
    grad2 = jax.jit(jax.grad(obj_fun, argnums=4), static_argnums=(0, 2))(
        model, key, n_obs, x_init, theta
    )
    assert_equal(grad1, grad2)


def test_simulate_models(
    model, model2, key, n_obs, x_init, theta, model_args, **kwargs
):
    """
    Test function to compare equivalent model definitions.

    Checks that equivalent model implementations produce the same simulation
    results for both measurements and latent states.

    Parameters
    ----------
    model : class
        First model class definition.
    model2 : class
        Second model class definition.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    model_args : dict
        Arguments to initialize the models.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """
    # instantiate the models
    model1 = model(**model_args)
    model2 = model2(**model_args)
    # simulate with each model
    y_meas1, x_state1 = pf.simulate(model1, key, n_obs, x_init, theta)
    y_meas2, x_state2 = pf.simulate(model2, key, n_obs, x_init, theta)
    assert_equal(y_meas1, y_meas2)
    assert_equal(x_state1, x_state2)


# --- loglik_full test functions -----------------------------------------------


def test_loglik_full_for(model, key, n_obs, x_init, theta, model_args, **kwargs):
    """
    Test function to compare old and new particle filter implementations.

    Compares the outputs of the old for-loop implementation and the vmap implementation
    of the particle filter for a set of test cases. Checks that all relevant outputs
    match to within numerical precision.

    Parameters
    ----------
    model : callable
        Model class or instance to use for the filter.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    model_args : dict
        Arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """
    # instantiate the model
    model = model(**model_args)
    # simulate without for-loop
    y_meas, x_state = pf.simulate(model, key, n_obs, x_init, theta)
    # joint loglikelihood with for-loop
    loglik1 = test.loglik_full_for(model, y_meas, x_state, theta)
    # joint loglikelihood with vmap
    loglik2 = pf.loglik_full(model, y_meas, x_state, theta)
    assert_equal(loglik1, loglik2)


def test_loglik_full_jit(model, key, n_obs, x_init, theta, model_args, **kwargs):
    """
    Test function to compare jitted and non-jitted loglikelihood and gradients.

    Checks that evaluating ``loglik_full`` and its gradients with and without
    JAX jit produces identical results.

    Parameters
    ----------
    model : callable
        The model class/constructor to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    model_args : dict
        Arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """
    # instantiate the model
    model = model(**model_args)
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # joint loglikelihood without jit
    loglik1 = pf.loglik_full(model, y_meas, x_state, theta)
    # joint loglikelihood with jit
    loglik_full_jit = jax.jit(pf.loglik_full, static_argnums=0)
    loglik2 = loglik_full_jit(model, y_meas, x_state, theta)
    assert_equal(loglik1, loglik2)
    # grad without jit
    grad1 = jax.grad(pf.loglik_full, argnums=(2, 3))(model, y_meas, x_state, theta)
    # grad with jit
    grad2 = jax.jit(jax.grad(pf.loglik_full, argnums=(2, 3)), static_argnums=0)(
        model, y_meas, x_state, theta
    )
    for i in range(2):
        assert_equal(grad1[i], grad2[i], context=f"index '{i}'")


def test_loglik_full_models(
    model, model2, key, n_obs, x_init, theta, model_args, **kwargs
):
    """
    Test function to compare joint loglikelihoods across two equivalent model
    implementations.

    Simulates data with model2 and checks that pf.loglik_full
    computed under both model1 and model2 agree.

    Parameters
    ----------
    model : class
        First model class definition.
    model2 : callable
        Second model class definition.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    model_args : dict
        Keyword arguments used to instantiate models.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """
    # instantiate the models
    model1 = model(**model_args)
    model2 = model2(**model_args)
    # simulate data
    y_meas, x_state = pf.simulate(
        model=model1, key=key, n_obs=n_obs, x_init=x_init, theta=theta
    )
    # joint loglikelihood with each model
    loglik1 = pf.loglik_full(model=model1, y_meas=y_meas, x_state=x_state, theta=theta)
    loglik2 = pf.loglik_full(model=model2, y_meas=y_meas, x_state=x_state, theta=theta)
    assert_equal(loglik1, loglik2)


# --- particle_filter test functions -------------------------------------------


def test_particle_filter_for(
    model, key, n_obs, x_init, theta, model_args, n_particles, **kwargs
):
    """
     Test function to compare old and new particle filter implementations.

     Compares the outputs of the old for-loop implementation and the new vmap
     implementation of the particle filter for a set of test cases.
     Checks that all relevant outputs match to within numerical precision.

     Parameters
     ----------
     model : callable
         The model class/constructor to instantiate.
     key : PRNGKey
         JAX random key.
     n_obs : int
         Number of observations.
     x_init : ndarray
         Initial latent state.
     theta : ndarray or dict
         Model parameters.
     n_particles : int
         Number of particles for the filter.
    model_args : dict
         Keyword arguments used to instantiate model.
    **kwargs : dict, optional
         Additional unused keyword arguments.
    """

    # Instantiate model
    model = model(**model_args)
    # define test cases
    test_cases = expand_grid(history=jnp.array([False, True]))
    n_cases = test_cases.shape[0]
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # old pf with for-loop
    key, subkey = random.split(key)
    pf_out1 = test.particle_filter_for(model, subkey, y_meas, theta, n_particles)
    for i in range(n_cases):
        case = test_cases.iloc[i]
        # new pf
        pf_out2 = pf.particle_filter(
            model, subkey, y_meas, theta, n_particles, score=False, fisher=False, **case
        )
        # check outputs
        if case["history"]:
            max_diff = {
                k: rel_err(pf_out1[k], pf_out2[k]) for k in ["x_particles", "logw"]
            }
            max_diff["ancestors"] = rel_err(
                X1=pf_out1["ancestors"], X2=pf_out2["resample_out"]["ancestors"]
            )
        else:
            max_diff = {
                k: rel_err(pf_out1[k][n_obs - 1], pf_out2[k])
                for k in ["x_particles", "logw"]
            }
            max_diff["loglik"] = rel_err(
                X1=test.particle_loglik(pf_out1["logw"]), X2=pf_out2["loglik"]
            )
        for k in max_diff.keys():
            assert_equal(max_diff[k], 0.0, context=f"key '{k}', case={case}")


def test_particle_filter_deriv(
    model, key, n_obs, x_init, theta, model_args, n_particles, **kwargs
):
    """
    Test function to check particle filter derivatives.

    Compares online and brute-force calculations of the score and Fisher information (hessian) for
    the particle filter, across multiple test cases. Asserts that both methods yield numerically
    identical results.

    Parameters
    ----------
    model : callable
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations.
    x_init : ndarray
        Initial state.
    theta : ndarray or dict
        Model parameters.
    model_args : dict
        Keyword arguments to initialize model.
    n_particles : int
        Number of particles for the particle filter.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate model
    model = model(**model_args)

    def accumulate_deriv(x_prev, x_curr, y_curr, theta):
        r"""
        Accumulator for both score and hessian.
        """
        grad_meas = jax.grad(model.meas_lpdf, argnums=2)
        grad_state = jax.grad(model.state_lpdf, argnums=2)
        hess_meas = jax.jacfwd(jax.jacrev(model.meas_lpdf, argnums=2), argnums=2)
        hess_state = jax.jacfwd(jax.jacrev(model.state_lpdf, argnums=2), argnums=2)
        alpha = grad_meas(y_curr, x_curr, theta) + grad_state(x_curr, x_prev, theta)
        beta = hess_meas(y_curr, x_curr, theta) + hess_state(x_curr, x_prev, theta)
        return (alpha, beta)

    # define test cases
    test_cases = expand_grid(
        history=jnp.array([False, True]),
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True]),
    )
    n_cases = test_cases.shape[0]
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # pf with history, no derivatives
    key, subkey = random.split(key)
    pf_out1 = pf.particle_filter(
        model,
        subkey,
        y_meas,
        theta,
        n_particles,
        score=False,
        fisher=False,
        history=True,
    )
    for i in range(n_cases):
        case = test_cases.iloc[i]
        # pf various history/derivatives
        pf_out2 = pf.particle_filter(model, subkey, y_meas, theta, n_particles, **case)
        # check outputs
        if case["history"]:
            max_diff = {
                k: rel_err(pf_out1[k], pf_out2[k]) for k in ["x_particles", "logw"]
            }
            max_diff["ancestors"] = rel_err(
                X1=pf_out1["resample_out"]["ancestors"],
                X2=pf_out2["resample_out"]["ancestors"],
            )
        else:
            max_diff = {
                k: rel_err(pf_out1[k][n_obs - 1], pf_out2[k])
                for k in ["x_particles", "logw"]
            }
        max_diff["loglik"] = rel_err(X1=pf_out1["loglik"], X2=pf_out2["loglik"])
        if case["score"] or case["fisher"]:
            # score and hess using smoothing accumulator
            x_particles = pf_out1["x_particles"]
            ancestors = pf_out1["resample_out"]["ancestors"]
            logw = pf_out1["logw"][n_obs - 1]
            alpha, beta = test.accumulate_smooth(
                logw=logw,
                x_particles=x_particles,
                ancestors=ancestors,
                y_meas=y_meas,
                theta=theta,
                accumulator=accumulate_deriv,
                mean=False,
            )
            prob = utils.logw_to_prob(logw)
            _score = jax.vmap(jnp.multiply)(prob, alpha)
            _hess = jax.vmap(lambda p, a, b: p * (jnp.outer(a, a) + b))(
                prob, alpha, beta
            )
            _score, _hess = jtu.tree_map(lambda x: jnp.sum(x, axis=0), (_score, _hess))
            max_diff["score"] = rel_err(_score, pf_out2["score"])
            if case["fisher"]:
                max_diff["fisher"] = rel_err(
                    -1.0 * (_hess - jnp.outer(_score, _score)), pf_out2["fisher"]
                )
        for k in max_diff.keys():
            assert_equal(max_diff[k], 0.0, context=f"key '{k}', case={case}")


# --- particle_filter_rb test functions ----------------------------------------


def test_particle_filter_rb_for(
    model, key, n_obs, x_init, theta, n_particles, model_args, **kwargs
):
    """
    Compare RB particle filter (vmap/scan) vs the for-loop reference.

    Runs both RB implementations across a small grid of derivative flags
    and checks that every reported output matches to numerical precision.

    Parameters
    ----------
    model : callable
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles for the filter.
    model_args : dict
        Keyword arguments used to instantiate model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate model
    model = model(**model_args)

    # define test cases
    test_cases = expand_grid(
        history=jnp.array([False]),  # save time by skipping True
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True]),
    )
    n_cases = test_cases.shape[0]
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    for i in range(n_cases):
        case = test_cases.iloc[i]
        key, subkey = random.split(key)
        # rb filter vmap
        pf_out1 = pf.particle_filter_rb(
            model, subkey, y_meas, theta, n_particles, **case
        )
        # rb filter for-loop
        pf_out2 = test.particle_filter_rb_for(
            model, subkey, y_meas, theta, n_particles, **case
        )
        # max_diff = jtu.tree_map(rel_err, pf_out1, pf_out2)
        # for k in max_diff.keys():
        #     assert_equal(max_diff[k], 0.0, context=f"key '{k}', case={case}")
        for k in pf_out1.keys():
            assert_equal(pf_out1[k], pf_out2[k], context=f"key='{k}', case={case}")


def test_particle_filter_rb_history(
    model, key, n_obs, x_init, theta, n_particles, model_args, **kwargs
):
    """
    Check that RB particle filter outputs are identical with and without history.

    Runs the RB filter twice per case (history=False vs history=True) and
    compares the reported outputs (loglik, score, fisher if requested).

    Parameters
    ----------
    model : callable
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate model
    model = model(**model_args)
    # define test cases
    test_cases = expand_grid(
        score=jnp.array([False, True]), fisher=jnp.array([False, True])
    )
    n_cases = test_cases.shape[0]
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    for i in range(n_cases):
        case = test_cases.iloc[i]
        key, subkey = random.split(key)
        # rb filter no history
        pf_out1 = pf.particle_filter_rb(
            model, subkey, y_meas, theta, n_particles, history=False, **case
        )
        # rb filter history
        pf_out2 = pf.particle_filter_rb(
            model, subkey, y_meas, theta, n_particles, history=True, **case
        )
        # check outputs
        keys = ["loglik"]
        keys = keys + ["score"] if case["score"] or case["fisher"] else keys
        keys = keys + ["fisher"] if case["fisher"] else keys
        max_diff = {k: rel_err(pf_out1[k], pf_out2[k]) for k in keys}
        for k in max_diff.keys():
            assert_equal(max_diff[k], 0.0, context=f"key '{k}', case={case}")


def test_particle_filter_rb_deriv(
    model, key, n_obs, x_init, theta, n_particles, model_args, **kwargs
):
    """
    Parameters
    ----------
    model : callable
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate model
    model = model(**model_args)

    # gradient and hessian functions
    def grad_step(x_curr, x_prev, y_curr, logw_prev, logw_aux, alpha_prev, beta_prev):
        """
        Update logw_targ, alpha, and beta.
        """
        logw_targ = (
            model.meas_lpdf(y_curr=y_curr, x_curr=x_curr, theta=theta)
            + model.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)
            + logw_prev
        )
        logw_prop = (
            model.step_lpdf(x_curr=x_curr, x_prev=x_prev, y_curr=y_curr, theta=theta)
            + logw_aux
        )
        alpha = (
            grad_state(x_curr, x_prev, theta)
            + grad_meas(y_curr, x_curr, theta)
            + alpha_prev
        )
        beta = (
            jnp.outer(alpha, alpha)
            + hess_meas(y_curr, x_curr, theta)
            + hess_state(x_curr, x_prev, theta)
            + beta_prev
        )
        return {
            "logw_targ": logw_targ,
            "logw_prop": logw_prop,
            "alpha": alpha,
            "beta": beta,
        }

    grad_meas = jax.grad(model.meas_lpdf, argnums=2)
    grad_state = jax.grad(model.state_lpdf, argnums=2)
    hess_meas = jax.jacfwd(jax.jacrev(model.meas_lpdf, argnums=2), argnums=2)
    hess_state = jax.jacfwd(jax.jacrev(model.state_lpdf, argnums=2), argnums=2)

    # define test cases
    test_cases = expand_grid(
        history=jnp.array([False, True]),
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True]),
    )
    n_cases = test_cases.shape[0]
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # pf with history, no derivatives
    key, subkey = random.split(key)
    pf_out1 = pf.particle_filter_rb(
        model,
        subkey,
        y_meas,
        theta,
        n_particles,
        score=False,
        fisher=False,
        history=True,
    )
    # brute-force derivative calculation
    # initialize
    n_theta = theta.size
    logw_prev = pf_out1["logw_bar"][0]
    alpha_prev = jnp.zeros((n_particles, n_theta))
    beta_prev = jnp.zeros((n_particles, n_theta, n_theta))
    loglik2 = jsp.special.logsumexp(logw_prev)
    # update for every observation
    for i_curr in range(1, n_obs):
        x_prev = pf_out1["x_particles"][i_curr - 1]
        x_curr = pf_out1["x_particles"][i_curr]
        y_curr = y_meas[i_curr]
        logw_aux = logw_prev
        # manual update calculation
        grad_full = jax.vmap(
            jax.vmap(grad_step, in_axes=(None, 0, None, 0, 0, 0, 0)),
            in_axes=(0, None, None, None, None, None, None),
        )(x_curr, x_prev, y_curr, logw_prev, logw_aux, alpha_prev, beta_prev)
        logw_curr = jax.vmap(
            lambda ltarg, lprop: jsp.special.logsumexp(ltarg)
            - jsp.special.logsumexp(lprop)
        )(grad_full["logw_targ"], grad_full["logw_prop"])
        loglik2 = loglik2 + jsp.special.logsumexp(logw_curr)
        alpha_curr = jax.vmap(pf.utils.tree_mean)(
            grad_full["alpha"], grad_full["logw_targ"]
        )
        beta_curr = jax.vmap(pf.utils.tree_mean)(
            grad_full["beta"], grad_full["logw_targ"]
        ) - jax.vmap(jnp.outer)(alpha_curr, alpha_curr)
        # set prev to curr
        logw_prev = logw_curr
        alpha_prev = alpha_curr
        beta_prev = beta_curr
    # finalize calculations
    loglik2 = loglik2 - n_obs * jnp.log(n_particles)
    gamma_curr = jax.vmap(lambda a, b: jnp.outer(a, a) + b)(alpha_curr, beta_curr)
    score2 = pf.utils.tree_mean(alpha_curr, logw_curr)
    fisher2 = pf.utils.tree_mean(gamma_curr, logw_curr) - jnp.outer(score2, score2)
    fisher2 = -1.0 * fisher2
    for i in range(n_cases):
        case = test_cases.iloc[i]
        # pf various history/derivatives
        pf_out2 = pf.particle_filter_rb(
            model, subkey, y_meas, theta, n_particles, **case
        )
        # check outputs
        max_diff = {"loglik": rel_err(pf_out2["loglik"], loglik2)}
        if case["score"] or case["fisher"]:
            max_diff["score"] = rel_err(pf_out2["score"], score2)
        if case["fisher"]:
            max_diff["fisher"] = rel_err(pf_out2["fisher"], fisher2)
        for k in max_diff.keys():
            assert_equal(max_diff[k], 0.0, context=f"key '{k}', case={case}")


# --- param_mwg_update test functions ------------------------------------------


def test_param_mwg_update_for(
    model, key, n_obs, x_init, theta, n_particles, model_args, **kwargs
):
    """
    Parameters
    ----------
    model : callable
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate model
    model = model(**model_args)
    # simulate without for-loop
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # mwg setup
    prior = mcmc.NormalDiagPrior(loc=theta, scale=jnp.abs(theta))
    rw_sd = jnp.array([0.1] * theta.size)
    # with default order
    theta_order = jnp.arange(theta.size)
    key, subkey = random.split(key)
    mwg_out1 = test.param_mwg_update_for(
        model, prior, subkey, theta, x_state, y_meas, rw_sd, theta_order
    )
    mwg_out2 = mcmc.param_mwg_update(
        model, prior, subkey, theta, x_state, y_meas, rw_sd, theta_order
    )
    for i in range(2):
        assert_equal(mwg_out1[i], mwg_out2[i], context=f"index {i}")
    # with non-default order
    key, subkey = random.split(key)
    n_updates = 10
    theta_order = random.choice(subkey, jnp.arange(theta.size), shape=(n_updates,))
    key, subkey = random.split(key)
    mwg_out1 = test.param_mwg_update_for(
        model, prior, subkey, theta, x_state, y_meas, rw_sd, theta_order
    )
    mwg_out2 = mcmc.param_mwg_update(
        model, prior, subkey, theta, x_state, y_meas, rw_sd, theta_order
    )
    for i in range(2):
        assert_equal(mwg_out1[i], mwg_out2[i], context=f"index {i}")


def test_param_mwg_update_jit(
    model, key, n_obs, x_init, theta, n_particles, model_args, **kwargs
):
    """
    Parameters
    ----------
    model : callable
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate model
    model = model(**model_args)
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # mwg setup
    prior = mcmc.NormalDiagPrior(loc=theta, scale=jnp.abs(theta))
    rw_sd = jnp.array([0.1] * theta.size)
    theta_order = jnp.arange(theta.size)
    # mwg update without jit
    key, subkey = random.split(key)
    mwg_out1 = mcmc.param_mwg_update(
        model, prior, subkey, theta, x_state, y_meas, rw_sd, theta_order
    )
    # mwg update with jit
    mwg_out2 = jax.jit(mcmc.param_mwg_update, static_argnums=(0, 1))(
        model, prior, subkey, theta, x_state, y_meas, rw_sd, theta_order
    )
    for i in range(2):
        assert_equal(mwg_out1[i], mwg_out2[i], context=f"index {i}")

    # objective function for gradient
    def obj_fun(model, prior, key, theta, x_state, y_meas, rw_sd, theta_order):
        theta_update, accept = mcmc.param_mwg_update(
            model, prior, key, theta, x_state, y_meas, rw_sd, theta_order
        )
        return jnp.sum(theta_update)

    # grad without jit
    grad1 = jax.grad(obj_fun, argnums=(3, 4, 5))(
        model, prior, subkey, theta, x_state, y_meas, rw_sd, theta_order
    )
    # grad with jit
    grad2 = jax.jit(jax.grad(obj_fun, argnums=(3, 4, 5)), static_argnums=(0, 1))(
        model, prior, subkey, theta, x_state, y_meas, rw_sd, theta_order
    )
    for i in range(3):
        assert_equal(grad1[i], grad2[i], context=f"index {i}")


# --- resample_mvn test functions ----------------------------------------------


def test_resample_mvn_for(key, **kwargs):
    """
    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    """

    n_particles = 25
    # define test cases
    test_cases = expand_grid(shape=[(), (1,), (1, 2), (2, 3, 4)])
    n_cases = test_cases.shape[0]
    # log-weights
    key, subkey = random.split(key)
    logw = random.normal(subkey, (n_particles,))
    for i in range(n_cases):
        case = test_cases.iloc[i]
        x_particles = jax.random.normal(
            key=subkey, shape=(n_particles,) + case["shape"]
        )
        # for-loop version
        new_particles1 = test.resample_mvn_for(
            key=subkey, x_particles_prev=x_particles, logw=logw
        )
        # vmap version
        new_particles2 = resamplers.resample_mvn(
            key=subkey, x_particles_prev=x_particles, logw=logw
        )
        for k in new_particles1.keys():
            assert_equal(
                new_particles1[k], new_particles2[k], context=f"key '{k}', case={case}"
            )


def test_resample_mvn_shape(key, **kwargs):
    """
    Check that shaped and flat particles give the same results.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    """

    n_particles = 25
    # define test cases
    test_cases = expand_grid(shape=[(), (1,), (1, 2), (2, 3, 4)])
    n_cases = test_cases.shape[0]
    # log-weights
    key, subkey = random.split(key)
    logw = random.normal(subkey, (n_particles,))
    for i in range(n_cases):
        case = test_cases.iloc[i]
        # original shape
        x_particles1 = jax.random.normal(
            key=subkey, shape=(n_particles,) + case["shape"]
        )
        new_particles1 = test.resample_mvn_for(
            key=subkey, x_particles_prev=x_particles1, logw=logw
        )
        # flattened shape
        dim2 = jnp.zeros(case["shape"]).size
        dim2 = (dim2,) if dim2 > 0 else ()
        x_particles2 = jax.random.normal(key=subkey, shape=(n_particles,) + dim2)
        new_particles2 = resamplers.resample_mvn(
            key=subkey, x_particles_prev=x_particles2, logw=logw
        )
        for k in ["mvn_mean", "mvn_cov"]:
            assert_equal(
                new_particles1[k], new_particles2[k], context=f"key '{k}', case={case}"
            )


def test_resample_mvn_jit(
    model, key, n_obs, x_init, theta, n_particles, model_args, **kwargs
):
    """
    Parameters
    ----------
    model : callable
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate model
    model = model(**model_args)

    # objective function for gradient
    def obj_fun(model, key, y_meas, theta, n_particles):
        out = pf.particle_filter(
            model, key, y_meas, theta, n_particles, resampler=resamplers.resample_mvn
        )
        return jtu.tree_reduce(
            lambda x, y: x + jnp.sum(y), pf.utils.rm_keys(out, "key"), jnp.array(0.0)
        )

    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # particle filter specification
    key, subkey = random.split(key)
    # pf without jit
    pf_out1 = pf.particle_filter(
        model, subkey, y_meas, theta, n_particles, resampler=resamplers.resample_mvn
    )
    # pf with jit
    pf_out2 = jax.jit(pf.particle_filter, static_argnums=(0, 4, 5))(
        model, subkey, y_meas, theta, n_particles, resampler=resamplers.resample_mvn
    )
    for k in pf_out1.keys():
        assert_equal(pf_out1[k], pf_out2[k]),
        f"failed at key '{k}'"
    # grad without jit
    grad1 = jax.grad(obj_fun, argnums=3)(model, key, y_meas, theta, n_particles)
    # grad with jit
    grad2 = jax.jit(jax.grad(obj_fun, argnums=3), static_argnums=(0, 4))(
        model, key, y_meas, theta, n_particles
    )
    assert_equal(grad1, grad2)


# --- particle_smooth test functions -------------------------------------------


def test_particle_smooth_for(
    model, key, n_obs, x_init, theta, n_particles, model_args, **kwargs
):
    """
    Parameters
    ----------
    model : callable
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate model
    model = model(**model_args)
    # simulate without for-loop
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # pf without for-loop
    key, subkey = random.split(key)
    pf_out = pf.particle_filter(model, subkey, y_meas, theta, n_particles, history=True)
    # pf_smooth with for-loop
    key, subkey = random.split(key)
    x_state1 = test.particle_smooth_for(
        key=subkey,
        logw=pf_out["logw"][n_obs - 1],
        x_particles=pf_out["x_particles"],
        ancestors=pf_out["resample_out"]["ancestors"],
    )
    # pf_smooth without for-loop
    x_state2 = pf.particle_smooth(
        key=subkey,
        logw=pf_out["logw"][n_obs - 1],
        x_particles=pf_out["x_particles"],
        ancestors=pf_out["resample_out"]["ancestors"],
    )
    assert_equal(x_state1, x_state2)


def test_particle_smooth_jit(
    model, key, n_obs, x_init, theta, n_particles, model_args, **kwargs
):
    """
    Parameters
    ----------
    model : callable
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate model
    model = model(**model_args)

    # objective function for gradient
    def obj_fun(model, key, y_meas, theta, n_particles):
        pf_out = pf.particle_filter(
            model, key, y_meas, theta, n_particles, history=True
        )
        return jnp.sum(
            pf.particle_smooth(
                key=subkey,
                logw=pf_out["logw"][n_obs - 1],
                x_particles=pf_out["x_particles"],
                ancestors=pf_out["resample_out"]["ancestors"],
            )
        )

    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # particle filter specification
    key, subkey = random.split(key)
    # pf without jit
    pf_out = pf.particle_filter(model, subkey, y_meas, theta, n_particles, history=True)
    # pf_smooth without jit
    key, subkey = random.split(key)
    x_state1 = pf.particle_smooth(
        key=subkey,
        logw=pf_out["logw"][n_obs - 1],
        x_particles=pf_out["x_particles"],
        ancestors=pf_out["resample_out"]["ancestors"],
    )
    # pf_smooth with jit
    x_state2 = jax.jit(pf.particle_smooth)(
        key=subkey,
        logw=pf_out["logw"][n_obs - 1],
        x_particles=pf_out["x_particles"],
        ancestors=pf_out["resample_out"]["ancestors"],
    )
    assert_equal(x_state1, x_state2)
    # grad without jit
    grad1 = jax.grad(obj_fun, argnums=3)(model, key, y_meas, theta, n_particles)
    # grad with jit
    grad2 = jax.jit(jax.grad(obj_fun, argnums=3), static_argnums=(0, 4))(
        model, key, y_meas, theta, n_particles
    )
    assert_equal(grad1, grad2)


# --- sde test functions -------------------------------------------------------


def test_sde_state_sample_for(
    model, model2, key, x_init, theta, n_particles, model_args, n_obs
):
    """
    Test function to compare sde implementations. Compares the outputs of
    two simulation approaches (for-loop vs scan/vmap) for the lv model.
    Note: state_sample_for() only implemented in test.lotvol_model.

    Parameters
    ----------
    model : class
        First model class definition.
    model2 : class
        Second model class definition.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the models.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """
    # instantiate both model variants
    model1 = model(**model_args)  # vmap version
    model2 = model2(**model_args)  # for-loop version
    n_res = model_args["n_res"]
    # generate previous timepoint
    key, subkey = random.split(key)
    x_prev = x_init
    x_prev = x_prev + random.normal(subkey, x_prev.shape)
    # simulate state using lax.scan
    x_state1 = model1.state_sample(key, x_prev, theta)
    # simulate state using
    x_state2 = model2.state_sample_for(key, x_prev, theta)
    assert_equal(x_state1, x_state2)


def test_sde_state_lpdf_for(
    model, model2, key, x_init, theta, n_particles, model_args, n_obs
):
    """
    Test function to compare sde lpdf implementations. Compares the outputs of
    two simulation approaches (for-loop vs scan/vmap) for the lv model.
    Note: state_sample_for() only implemented in test.lotvol_model.

    Parameters
    ----------
    model : class
        First model class definition.
    model2 : class
        Second model class definition.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the model.
    n_obs : int
        Number of observations to simulate.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate both model variants
    model1 = model(**model_args)  # vmap version
    model2 = model2(**model_args)  # for-loop version
    n_res = model_args["n_res"]
    # generate previous timepoint
    key, subkey = random.split(key)
    x_prev = x_init
    x_prev = x_prev + random.normal(subkey, x_prev.shape)
    # simulate state using lax.scan
    x_curr = model1.state_sample(key, x_prev, theta)
    # lpdf using vmap
    lp1 = model1.state_lpdf(x_curr, x_prev, theta)
    # lpdf using for-loop
    lp2 = model2.state_lpdf_for(x_curr, x_prev, theta)
    assert_equal(lp1, lp2)


def test_bridge_step_for(
    model, key, n_obs, x_init, theta, n_particles, model_args, **kwargs
):
    """
    Parameters
    ----------
    model : class
        Model class to instantiate.
    key : PRNGKey
        JAX random key.
    n_obs : int
        Number of observations to simulate.
    x_init : ndarray
        Initial latent state.
    theta : ndarray or dict
        Model parameters.
    n_particles : int
        Number of particles to use.
    model_args : dict
        Keyword arguments to initialize the model.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    # instantiate models
    model = model(**model_args)
    n_res = model_args["n_res"]
    # generate previous timepoint
    key, subkey = random.split(key)
    x_prev = jnp.block([[jnp.zeros((n_res - 1, 2))], [jnp.log(jnp.array([5.0, 3.0]))]])
    y_curr = jnp.exp(x_prev[-1]) + theta[6:8] * random.normal(
        subkey, (x_prev.shape[1],)
    )
    # bridge proposal using lax.scan
    x_curr1, logw1 = model.bridge_step(
        key=key,
        x_prev=x_prev,
        y_curr=y_curr,
        theta=theta,
        Y=jnp.log(y_curr),
        A=jnp.eye(2),
        Omega=jnp.eye(2),
    )
    # bridge proposal using for
    x_curr2, logw2 = model._bridge_step_for(
        key=key,
        x_prev=x_prev,
        y_curr=y_curr,
        theta=theta,
        Y=jnp.log(y_curr),
        A=jnp.eye(2),
        Omega=jnp.eye(2),
    )
    assert_equal(x_curr1, x_curr2)
    assert_equal(logw1, logw2)


# --- resample_ot tests --------------------------------------------------------


def test_resample_ot_sinkhorn(key, n_particles, n_dim, **kwargs):
    """
    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    n_dim : int
        Number of dimensions to simulate.
    n_particles : int
        Number of particles to use.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """
    n_iter_ot = 1000
    n_iter_custom = 10_000
    epsilon = 0.1
    key, *subkeys = random.split(key, 5)
    a = random.dirichlet(subkeys[0], alpha=jnp.ones((n_particles,)))
    b = random.dirichlet(subkeys[1], alpha=jnp.ones((n_particles,)))
    # NOTE: u and v must be 2d arrays, with 1st dim the number of points
    u = random.normal(subkeys[2], shape=(n_particles, n_dim))
    v = random.normal(subkeys[3], shape=(n_particles, n_dim))
    test_cases = expand_grid(
        method=["jax-ott", "resample_ot"], scaled=jnp.array([False, True])
    )
    n_cases = test_cases.shape[0]
    sinkhorn_custom = jax.jit(test.sinkhorn_test, static_argnames="n_iterations")
    x_over_y = jax.vmap(lambda x, y: x / y)  # to divide matrix by vector

    for i in range(n_cases):
        case = test_cases.iloc[i]
        if case["scaled"]:
            scale_cost = jnp.max(jnp.var(u, axis=0))
            scale_cost = n_dim * scale_cost
        else:
            scale_cost = 1.0
        sinkhorn_kwargs = {"min_iterations": n_iter_ot, "max_iterations": n_iter_ot}
        pointcloud_kwargs = {"epsilon": epsilon, "scale_cost": 1.0}
        custom_kwargs = {
            "epsilon": epsilon,
            "scale_cost": scale_cost,
            "n_iterations": n_iter_custom,
        }
        if case["method"] == "jax-ott":
            custom_a = a
            custom_v = v
            custom_kwargs.update({"a": custom_a, "u": u, "b": b, "v": custom_v})
            # sinkhorn with jax-ott
            geom = pointcloud.PointCloud(
                u / jnp.sqrt(scale_cost), v / jnp.sqrt(scale_cost), **pointcloud_kwargs
            )
            problem = linear_problem.LinearProblem(geom, a=a, b=b)
            solver = sinkhorn.Sinkhorn(**sinkhorn_kwargs)
            sink = solver(problem)
            # sink = sinkhorn.sinkhorn(geom, a, b,
            #                          **sinkhorn_kwargs)
            out1 = {
                "P": sink.matrix,
                "tsp": x_over_y(sink.apply(inputs=v.T, axis=1).T, a),
            }
        elif case["method"] == "resample_ot":
            custom_a = jnp.ones(n_particles) / n_particles
            custom_v = u
            custom_kwargs.update({"a": custom_a, "u": u, "b": a, "v": custom_v})
            # sinkhorn with resample-ott
            # kwargs need to be jitted each time,
            # can't make dict static argument
            resampler = partial(
                pf.particle_resamplers.resample_ot,
                scaled=case["scaled"],
                sinkhorn_kwargs=sinkhorn_kwargs,
                pointcloud_kwargs=pointcloud_kwargs,
            )
            out1 = jax.jit(resampler)(
                x_particles_prev=u,
                logw=jnp.log(a) - 5.0,
                key=key,
            )
            out1["P"] = out1["sink"].matrix
            out1["tsp"] = out1["x_particles"]
        # sinkhorn with custom code
        _, _, P2, C2 = sinkhorn_custom(**custom_kwargs)
        out2 = {
            "P": P2,
            # note: using P1 instead since
            # transport errors propagate quite a bit
            "tsp": x_over_y(jnp.matmul(out1["P"], custom_v), custom_a),
        }
        for k in out2.keys():  # Note: out1 has different keys!
            err = rel_err(out1[k], out2[k])
            assert err < 0.001, f"failed at key '{k}', case '{case}'"


def test_resample_ot_jit(key, n_particles, **kwargs):
    """
    Test various ways of jitting arguments.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    n_particles : int
        Number of particles to use.
    **kwargs : dict, optional
        Additional unused keyword arguments.
    """

    n_res = 5
    n_dim = 2
    n_iterations = 1000
    epsilon = jnp.array(0.1)
    key, *subkeys = random.split(key, 3)
    prob = random.dirichlet(subkeys[0], alpha=jnp.ones((n_particles,)))
    logw = jnp.log(prob) - 5.0
    x_particles_prev = random.normal(subkeys[1], shape=(n_particles, n_res, n_dim))
    test_cases = expand_grid(
        method=["jitted_1", "jitted_2"],
        kwargs=jnp.array([False, True]),
        scaled=jnp.array([False, True]),
    )
    n_cases = test_cases.shape[0]

    for i in range(n_cases):
        case = test_cases.iloc[i]
        if case["scaled"]:
            # scale_cost = jax.vmap(jnp.var,
            #                       in_axes=1)(x_particles_prev)
            scale_cost = jnp.max(jnp.var(x_particles_prev, axis=0))
            scale_cost = n_dim * n_res * scale_cost
        else:
            scale_cost = 1.0
        if case["kwargs"]:
            sinkhorn_kwargs = {
                "min_iterations": n_iterations,
                "max_iterations": n_iterations,
            }
            pointcloud_kwargs = {"epsilon": epsilon, "scale_cost": 1.0}
        else:
            sinkhorn_kwargs = {}
            pointcloud_kwargs = {}
        # unjitted
        resampler = partial(
            pf.particle_resamplers.resample_ot,
            scaled=case["scaled"],
            sinkhorn_kwargs=sinkhorn_kwargs,
            pointcloud_kwargs=pointcloud_kwargs,
        )
        out1 = resampler(x_particles_prev=x_particles_prev, logw=logw, key=key)
        # jitted
        if case["method"] == "jitted_1":
            resampler = jax.jit(resampler)
            out2 = resampler(x_particles_prev=x_particles_prev, logw=logw, key=key)
        elif case["method"] == "jitted_2":

            @jax.jit
            def resampler(x_particles_prev, logw, key):
                return pf.particle_resamplers.resample_ot(
                    x_particles_prev=x_particles_prev,
                    logw=logw,
                    key=key,
                    scaled=case["scaled"],
                    sinkhorn_kwargs=sinkhorn_kwargs,
                    pointcloud_kwargs=pointcloud_kwargs,
                )

            out2 = resampler(x_particles_prev=x_particles_prev, logw=logw, key=key)
        # jitted vs unjitted
        for k in ["x_particles"]:
            err = rel_err(out1[k], out2[k])
            assert err < 1e-4, f"failed at key '{k}', case '{case}'"


# def test_for_particle_filter(self):
#     # un-self setUp members
#     key = self.key
#     theta = self.theta
#     x_init = self.x_init
#     model_args = self.model_args
#     n_obs = self.n_obs
#     n_particles = self.n_particles
#     model = self.Model(**model_args)
#     # simulate without for-loop
#     key, subkey = random.split(key)
#     y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
#     # particle filter specification
#     key, subkey = random.split(key)
#     # pf with for-loop
#     pf_out1 = test.particle_filter_for(model, subkey,
#                                        y_meas, theta, n_particles)
#     # pf without for-loop
#     pf_out2 = pf.particle_filter(
#         model, subkey, y_meas, theta, n_particles)
#     for k in pf_out1.keys():
#         with self.subTest(k=k):
#             self.assert_equal(pf_out1[k], pf_out2[k])


# def test_jit_particle_filter(self):
#     # un-self setUp members
#     key = self.key
#     theta = self.theta
#     x_init = self.x_init
#     model_args = self.model_args
#     n_obs = self.n_obs
#     n_particles = self.n_particles
#     model = self.Model(**model_args)
#     # simulate data
#     key, subkey = random.split(key)
#     y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
#     # particle filter specification
#     key, subkey = random.split(key)
#     # pf without jit
#     pf_out1 = pf.particle_filter(
#         model, subkey, y_meas, theta, n_particles)
#     # pf with jit
#     pf_out2 = jax.jit(pf.particle_filter, static_argnums=(0, 4))(
#         model, subkey, y_meas, theta, n_particles)
#     for k in pf_out1.keys():
#         with self.subTest(k=k):
#             self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)

#     # objective function for gradient
#     def obj_fun(model, key, y_meas, theta, n_particles):
#         return pf.particle_loglik(pf.particle_filter(
#             model, key, y_meas, theta, n_particles)["logw"])
#     # grad without jit
#     grad1 = jax.grad(obj_fun, argnums=3)(
#         model, key, y_meas, theta, n_particles)
#     # grad with jit
#     grad2 = jax.jit(jax.grad(obj_fun, argnums=3), static_argnums=(0, 4))(
#         model, key, y_meas, theta, n_particles)
#     self.assertAlmostEqual(rel_err(grad1, grad2), 0.0)


# def test_models_particle_filter(self):
#     # un-self setUp members
#     key = self.key
#     theta = self.theta
#     x_init = self.x_init
#     model_args = self.model_args
#     n_obs = self.n_obs
#     n_particles = self.n_particles
#     model1 = self.Model(**model_args)
#     model2 = self.Model2(**model_args)
#     # simulate with inherited class
#     y_meas, x_state = pf.simulate(model2, key, n_obs, x_init, theta)
#     # particle filter specification
#     key, subkey = random.split(key)
#     # pf with non-inherited class
#     pf_out1 = pf.particle_filter(
#         model1, subkey, y_meas, theta, n_particles)
#     # pf with inherited class
#     pf_out2 = pf.particle_filter(
#         model2, subkey, y_meas, theta, n_particles)
#     for k in pf_out1.keys():
#         with self.subTest(k=k):
#             self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)
