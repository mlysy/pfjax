# parameter values
mu = 5
sigma = 1
tau = .1
theta = np.array([mu, sigma, tau])

# data specification
dt = .1
n_obs = 5
x_init = np.array([0.])

# simulate data
y_meas, x_state = meas_sim(n_obs, x_init, theta)

print("y_meas = \n", y_meas)
print("x_state = \n", x_state)

n_particles = 7
pf_out = particle_filter(y_meas, theta, n_particles)
pf_out = particle_filter(y_meas, theta, n_particles)
pf_out = particle_filter(y_meas, theta, n_particles)

print("pf_out = \n", pf_out)

# calculate marginal loglikelihood
pf_loglik = particle_loglik(pf_out["logw_particles"])

print("pf_loglik = \n", pf_loglik)

# sample from posterior `p(x_{0:T} | y_{0:T}, theta)`
n_sample = 11
X_state = particle_smooth(
    pf_out["logw_particles"][n_obs-1],
    pf_out["X_particles"],
    pf_out["ancestor_particles"],
    n_sample
)

print("X_state = \n", X_state)
