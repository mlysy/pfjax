import time
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
sns.set()

import jax
import jax.numpy as jnp
import jax.random as random

import pfjax as pf
from pfjax import particle_resamplers
# from pfjax.loglik_full import loglik_full
from pfjax import models
import optax
from functools import partial
import projplot as pjp

import warnings
warnings.filterwarnings('ignore')

from pfjax.experimental.models import *
from pfjax import experimental

key = random.PRNGKey(0)

_theta = np.array([0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1])
_noise = 0.5
tau = np.array([1,1,1,1]) * _noise
theta = np.concatenate([_theta, tau])

dt = 0.1
n_res = 1
n_obs = 5

pgnet_bridge = models.PGNETModel(dt, n_res, bootstrap=False)
pgnet_no_dna = PGNETModelNoDNA(dt, n_res, bootstrap=False)

key, subkey = random.split(key)

x_init = jnp.block([[jnp.zeros((n_res-1, 4))],
                   [jnp.log(jnp.array([8., 8., 8., 5.]))]])

y_meas, x_state = pf.simulate(pgnet_bridge, subkey, n_obs, x_init, theta)

tau_unobserved = np.array([1,1,1]) * _noise
theta_unobserved = np.concatenate([_theta, tau_unobserved, np.array([np.log(5.)])])

x_init, logw = pgnet_no_dna.pf_init(key, y_meas[0, :3], theta_unobserved)
print(x_init.shape)
print("DNA obsreved:\n ")

x_init2, logw = pgnet_bridge.pf_init(key, y_meas[0], theta)

print("DNA unobserved PF: \n")

# print(pgnet_no_dna.pf_step(key, x_init, y_meas[1, :3], theta_unobserved))

temp = pf.particle_filter(
        theta=theta_unobserved, 
        model=pgnet_no_dna, 
        y_meas=y_meas[:, :3], 
        n_particles=1e4, 
    key=key,
        resampler = particle_resamplers.resample_mvn)

print(temp)

print("DNA observed PF: \n")

# print(pgnet_bridge.pf_step(key, x_init2, y_meas[0], theta))

temp = pf.particle_filter(
        theta=theta,
        model=pgnet_bridge, 
        y_meas=y_meas, 
        n_particles=250, 
    key=key,
        resampler = particle_resamplers.resample_mvn)

print(temp)
