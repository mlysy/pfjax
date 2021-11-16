key = random.PRNGKey(0)

# parameter values
mu = 5
sigma = 1
tau = .1
theta = jnp.array([mu, sigma, tau])

print(theta)

# data specification
dt = .1
n_obs = 5
x_init = jnp.array([0.])

# simulate data
key, subkey = random.split(key)
y_meas, x_state = meas_sim(n_obs, x_init, theta, subkey)

print("y_meas = \n", y_meas)
print("x_init = \n", x_init)
print("x_state = \n", x_state)

# run particle filter
n_particles = 7
key, subkey = random.split(key)
particle_filter_jitted = jax.jit(particle_filter, static_argnums=(2,))
pf_out = particle_filter_jitted(y_meas, theta, n_particles, subkey)

print("pf_out = \n", pf_out)
