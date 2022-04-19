import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as random

import pfjax as pf
from pfjax import models
import optax
from functools import partial


# parameter values
alpha = 1.0
beta = 1.0
gamma = 4.0
delta = 1.0
sigma_h = 0.1
sigma_l = 0.1
tau_h = 0.25
tau_l = 0.25

theta = np.array([alpha, beta, gamma, delta, sigma_h, sigma_l, tau_h, tau_l])
theta_names = ["alpha", "beta", "gamma", "delta",
               "sigma_h", "sigma_l", "tau_h", "tau_l"]
theta_lims = np.array(list(zip(theta - (theta/2), theta + (theta/2))))

dt = 0.1
n_res = 1 
n_obs = 100

key = random.PRNGKey(0)
key, subkey = random.split(key)

lotvol_model = models.LotVolModelLog(dt, n_res, bootstrap = False) 
theta = jnp.log(theta)

x_init = jnp.block([[jnp.zeros((n_res-1, 2))],
                    [jnp.log(jnp.array([5., 3.]))]])

y_meas, x_state = pf.simulate(model = lotvol_model, 
                              n_obs = n_obs, 
                              x_init = x_init, 
                              theta = theta, 
                              key = subkey)

ret = pf.particle_filter(lotvol_model, key, y_meas, theta, 100, 
                         particle_sampler=pf.particle_resample_mvn)

print(ret)