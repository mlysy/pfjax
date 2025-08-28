import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import pfjax as pf
import pfjax.experimental.models

# import pfjax.models
# import pfjax.sde
# import lotvol_model as lv


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
n_res = 10
n_obs = 7
x_init = jnp.block([[jnp.zeros((n_res - 1, 2))], [jnp.log(jnp.array([5.0, 3.0]))]])
# simulate with inherited class
lv_model = pf.experimental.models.LotVolModel(dt=dt, n_res=n_res)
y_meas, x_state = pf.simulate(lv_model, key, n_obs, x_init, theta)

# # bridge proposal
# lv_model.bridge_step(
#     key=key,
#     x_prev=x_state[0],
#     Y=jnp.log(y_meas[1]),
#     theta=theta,
#     A=jnp.eye(2),
#     Omega=jnp.eye(2)
# )


# simulate with inherited class
lv_model1 = pf.LotVolModel(dt=dt, n_res=n_res)
y_meas1, x_state1 = pf.simulate(lv_model1, key, n_obs, x_init, theta)
# simulate with non-inherited class
lv_model2 = lv.LotVolModel(dt=dt, n_res=n_res)
y_meas2, x_state2 = pf.simulate(lv_model2, key, n_obs, x_init, theta)

y_meas1 - y_meas2
x_state1 - x_state2

n_state = (n_res, 2)

key = random.PRNGKey(0)
lv_model = pf.LotVolModel(dt=dt, n_res=n_res)

key, subkey = random.split(key)
x_prev = random.normal(subkey, n_state)

key, subkey = random.split(key)
x_curr = lv_model.state_sample(x_prev, theta, key)
x_curr_for = lv_model.state_sample_for(x_prev, theta, key)

print("x_curr - x_curr_for = \n", x_curr - x_curr_for)

state_lp = lv_model.state_lpdf(x_curr, x_prev, theta)
state_lp_for = lv_model.state_lpdf_for(x_curr, x_prev, theta)
print("state_lp - state_lp_for= \n", state_lp - state_lp_for)

# --- particle filter ----------------------------------------------------------

pf.particle_filter(
    lv_model,
    key,
    y_meas,
    theta,
    n_particles=5,
    particle_sampler=pf.particle_resample_ot,
)

key, subkey = random.split(key)
x_init = init_sample(
    y_init=jnp.log(jnp.array([5.0, 3.0])),
    theta=jnp.append(theta[0:6], jnp.array([0.0, 0.0])),
    key=subkey,
)

n_obs = 100
key, subkey = random.split(key)
y_meas, x_state = meas_sim(n_obs, x_init, theta, subkey)
t_seq = jnp.arange(n_obs) * dt

plt.plot(t_seq, y_meas[:, 0])
# plt.show()

n_particles = 100
pf_out = particle_filter(y_meas, theta, n_particles, key)
