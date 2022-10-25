---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Comparison of Gradient and Hessian Estimation Algorithms

**Martin Lysy -- University of Waterloo**

**September 15, 2022**

+++

## Summary

For parameter inference with state-space models, particle filters are useful not only for estimating the marginal loglikelihood $\ell(\tth) = \log p(\yy_{0:T} \mid \tth)$, but also its gradient and hessian functions, $\nabla \ell(\tth) = \frac{\partial}{\partial \tth} \ell(\tth)$ and $\nabla^2 \ell(\tth) = \frac{\partial^2}{\partial \tth \partial \tth'} \ell(\tth)$.

This module compares the speed and accuracy of various particle filter algorithms for the latter.  In particular, let $N$ denote the number of particles and $T$ denote the number of observations.  The particle filter based gradient and hessian algorithms to be compared here are:

1.  Automatic differentiation through the "basic" particle filter loglikelihood, i.e., the algorithm described in Section 4.1 of {cite:t}`doucet.johansen09` with multinomial resampler `pfjax.particle_resamplers.resample_multinomial()`.  This algorithm scales as $\bO(NT)$ but is known to produce biased results {cite:p}`corenflos.etal21`.  

2.  A modified version of the basic particle filter {cite:p}`cappe.moulines05` of which the bi-product are estimates of $\nabla \ell(\tth)$ and $\nabla^2 \ell(\tth)$.  This algorithm is unbiased and scales as $\bO(NT)$, but the variance of the estimates scales as $\bO(T^2/N)$ {cite:p}`poyiadjis.etal11`.  In other words, the number of particles $N$ must increase at least quadratically with the number of observations $T$ to keep the variance of the gradient and hessian estimators bounded.

3.  A bi-product of the "Rao-Blackwellized" (RB) particle filter developped by {cite:t}`poyiadjis.etal11` (also with multinomial resampling).  This algorithm also estimates the gradient and hessian unbiasedly.  Its computational complexity is $\bO(N^2 T)$, but the variance of the gradient/hessian estimates is $\bO(T/N)$ {cite:p}`poyiadjis.etal11`.

### Benchmark Model

We'll be using a Bootstrap filter for the Brownian motion with drift model defined in the [Introduction](pfjax.ipynb):

$$
\begin{aligned}
x_0 & \sim \N(0, \sigma^2 \dt) \\
x_t & \sim \N(x_{t-1} + \mu \dt, \sigma^2 \dt) \\
y_t & \sim \N(x_t, \tau^2),
\end{aligned}
$$

where the model parameters are $\tth = (\mu, \sigma, \tau)$.  The details of setting up the appropriate model class are provided in the [Introduction](pfjax.ipynb).  Here we'll use the version of this model provided with **PFJAX**: `pfjax.models.BMModel`.

### Methods to be Added to the Comparisons

4.  Automatic differentiation through a particle filter with multivariate normal resampling scheme `pfjax.particle_resamplers.resample_mvn()`.  This method is extremely fast and accurate as long as $p(\xx_{t} \mid \yy_{0:t-1}, \tth)$ is well-approximated by a multivariate normal.  It should probably be included for comparison, though since $p(\xx_{t} \mid \yy_{0:t-1}, \tth)$ is exactly Gaussian here its results are likely to be overly optimistic.

5.  Automatic differentiation through a particle filter with optimal transport resampling scheme `pfjax.particle_resamples.resample_ot()` proposed by {cite:t}`corenflos.etal22`.  This method is unbiased (at least for large $N$) and its computations scale as $\bO(N^2 T)$.  However, the underlying optimal transport algorithm as implemented by the **ott-jax** package requiest careful tuning to be of comparable speed to any of the algorithms 1-3 above.  

<!-- In this module we demontrate how to maximize the marginal likelihood for the state-space model,

$$
\begin{aligned}
\Ell(\tth) & = p(\yy_{0:T} \mid \tth) \\
& = \int \prod_{t=0}^T p(\yy_t \mid \xx_t, \tth) \times \prod_{t=1}^T p(\xx_t \mid \xx_{t-1}, \tth) \times p(\xx_0 \mid \tth)\, d \xx_{0:T},   
\end{aligned}
$$

using a particle filter approximation for $\Ell(\tth)$.  In particular:

- We obtain particle filter estimates of $\ell(\tth) = \log \Ell(\tth)$ and $\nabla \ell(\tth)$ in order to conduct stochastic optimization in search of $\hat \tth = \argmax_{\tth} \ell(\tth)$.

- We obtain a particle filter estimate of $\nabla^2 \ell(\tth)$ to be used for variance estimation, e.g., for calculating standard errors. -->

<!-- ## Example: Brownian Motion with Drift

The model is
$$
\begin{aligned}
x_0 & \sim \pi(x_0) \propto 1 \\
x_t & \sim \N(x_{t-1} + \mu \dt, \sigma^2 \dt) \\
y_t & \sim \N(x_t, \tau^2),
\end{aligned}
$$

with $\tth = (\mu, \sigma, \tau)$.  Note that with $\pi(x_0) \propto 1$, we may condition on $y_0$ and obtain $x_0 \mid y_0 \sim \N(y_0, \tau^2)$.  In this case, the marginal likelihood is defined as 

$$
\Ell(\tth) = p(\yy_{1:T} \mid y_0, \tth)
$$

rather than by $p(\yy_{0:T} \mid \tth)$.  The reason is that the latter expression requires one to integrate over $x_0 \sim \pi(x_0)$, which can only be done when $\pi(x_0)$ is a proper prior.  However, for our choice of $\pi(x_0) \propto 1$ this is not the case.  On the other hand, $p(\yy_{1:T} \mid y_0, \tth)$ only requires us to integrate over $p(x_0 \mid y_0, \tth)$, which only requires this distribution to be proper, as is the case here.

Brownian motion with drift is a simple model that will serve as a benchmark for further experiments. Note that in this particular model, we have access to the closed form of $\Ell(\tth)$ (see below for derivation). Since our method is an approximation method, it will serve as a reasonable sanity check and debugging tool to ensure that our method is not only technically sound, but also empirically verifiable. -->

```{code-cell} ipython3
# jax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random
from functools import partial
# plotting
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import projplot as pjp
# pfjax
import pfjax as pf
from pfjax.models import BMModel
```

## Simulate Data

```{code-cell} ipython3
# parameter values
mu = .1
sigma = .2
tau = .1
theta_true = jnp.array([mu, sigma, tau])

# data specification
dt = .5
n_obs = 100
x_init = jnp.array(0.)

# initial key for random numbers
key = jax.random.PRNGKey(0)

# simulate data
bm_model = BMModel(dt=dt)
key, subkey = jax.random.split(key)
y_meas, x_state = pf.simulate(
    model=bm_model,
    key=subkey,
    n_obs=n_obs,
    x_init=x_init,
    theta=theta_true
)

# plot data
plot_df = (pd.DataFrame({"time": jnp.arange(n_obs) * dt,
                         "x_state": jnp.squeeze(x_state),
                         "y_meas": jnp.squeeze(y_meas)})
           .melt(id_vars="time", var_name="type"))
sns.relplot(
    data=plot_df, kind="line",
    x="time", y="value", hue="type"
)
```

## Loglikelihood Comparisons

Before checking derivatives, let's start by comparing the speed and accuracy of the underlying particle filters, namely, the $\bO(NT)$ complexity algorithm of `pfjax.particle_filter()` and the $\bO(N^2T)$ algorithm of `pfjax.particle_filter_rb()`.  Accuracy is assessed visually using projection plots as described in the [Introduction](pfjax.ipynb).

Note that both `bm_loglik_basic()` and `bm_loglik_rb()` below are internally vectorized over multiple values of $\tth$, with each given a separate random seed.

```{code-cell} ipython3
def bm_loglik_exact(theta, y_meas):
    """
    Exact loglikelihood of the BM model.
    """
    theta = jnp.atleast_2d(theta)
    ll = jax.vmap(lambda _theta: bm_model.loglik_exact(
        y_meas=y_meas, 
        theta=_theta
    ))(theta)
    return jnp.squeeze(ll)

def bm_loglik_basic(theta, y_meas, key, n_particles):
    """
    Basic particle filter approximation of the loglikelihood.
    """
    theta = jnp.atleast_2d(theta)
    subkeys = jax.random.split(key, num=theta.shape[0])
    ll = jax.vmap(lambda _theta, _key: pf.particle_filter(
        model=bm_model,
        key=_key,
        y_meas=y_meas,
        n_particles=n_particles,
        theta=_theta,
        history=False,
        score=False,
        fisher=False
    )["loglik"])(theta, subkeys)
    return jnp.squeeze(ll)

def bm_loglik_rb(theta, y_meas, key, n_particles):
    """
    RB particle filter approximation of the loglikelihood.
    """
    theta = jnp.atleast_2d(theta)
    subkeys = jax.random.split(key, num=theta.shape[0])
    ll = jax.vmap(lambda _theta, _key: pf.particle_filter_rb(
        model=bm_model,
        key=_key,
        y_meas=y_meas,
        n_particles=n_particles,
        theta=_theta,
        history=False,
        score=False,
        fisher=False
    )["loglik"])(theta, subkeys)
    return jnp.squeeze(ll)
```

### Timing Comparisons

Let's first jit-compile a simplified version of the loglikelihoods for the projection plots and compare timings.

```{code-cell} ipython3
# jit-compiled exact loglikelihood (timed for reference)
bm_ll_exact = jax.jit(partial(bm_loglik_exact, y_meas=y_meas))

%timeit bm_ll_exact(theta_true)

# jit-compiled basic particle filter
n_particles_basic = 2500
key, subkey = jax.random.split(key)
bm_ll_basic = jax.jit(partial(bm_loglik_basic, y_meas=y_meas,
                             n_particles=n_particles_basic, key=subkey))

%timeit bm_ll_basic(theta_true)

# jit-compiled RB particle filter
n_particles_rb = 400
key, subkey = jax.random.split(key)
bm_ll_rb = jax.jit(partial(bm_loglik_rb, y_meas=y_meas,
                             n_particles=n_particles_rb, key=subkey))

%timeit bm_ll_rb(theta_true)
```

### Projection Plots

```{code-cell} ipython3
# projection plot specification
n_pts = 100 # number of evaluation points per plot
theta_lims = jnp.array([[-.5, .5], [.1, .4], [.05, .2]])  # plot limits for each parameter
theta_names = ["mu", "sigma", "tau"] # parameter names

# projection plots for exact loglikelihood
df_exact = pjp.proj_plot(
    fun=bm_ll_exact,
    x_opt=theta_true, 
    x_lims=theta_lims, 
    x_names=theta_names, 
    n_pts=n_pts,
    vectorized=True,
    plot=False
)

# projection plots for basic particle filter
df_basic = pjp.proj_plot(
    fun=bm_ll_basic,
    x_opt=theta_true, 
    x_lims=theta_lims, 
    x_names=theta_names, 
    n_pts=n_pts,
    vectorized=True,
    plot=False
)

# projection plots for RB particle filter
df_rb = pjp.proj_plot(
    fun=bm_ll_rb,
    x_opt=theta_true, 
    x_lims=theta_lims, 
    x_names=theta_names, 
    n_pts=n_pts,
    vectorized=True,
    plot=False
)

#merge data frames and plot them
plot_df = pd.concat([df_exact, df_basic, df_rb], ignore_index=True)
plot_df["method"] = np.repeat(["exact", "pf_basic", "pf_rb"], len(df_exact["variable"]))
rp = sns.relplot(
    data=plot_df, kind="line",
    x="x", y="y", 
    hue="method",
    col="variable",
    col_wrap = 3,
    facet_kws=dict(sharex=False, sharey=False)
)
rp.set_titles(col_template="{col_name}")
rp.set(xlabel=None)
rp.set(ylabel="loglikelihood")
# add true parameter values
for ax, theta in zip(rp.axes.flat, theta_true):
    ax.axvline(theta, linestyle="--", color="black")
```

**Conclusions:** 

- We used 2500 particles for the standard filter but only 400 particles for the RB filter.  The latter takes about 5x longer to compute but supposedly has lower variance.  In this particular case this does not seem to hold, i.e., the RB filter takes longer and appears to be more variable.  This suggests that the primary use of the RB filter is for calculating accurate gradients, as we shall see below.

- Both particle filters reasonably approximate $\ell(\tth)$ when $\mu$ is at its true value.  They don't do as well when $\mu$ is far from its true value.  This is likely due to particle degeneracy in that case.

+++

## Gradient Calculations

Here we'll check the three gradient algorithms: 

1. `auto`: Automatic differentiation through the multinomial sampler (known to be biased).
2. `acc`: The "accumulator" method of {cite:t}`cappe.moulines05` through the basic particle filter (unbiased but high variance).
3. `rb`: The method of {cite:t}`poyiadjis.etal11` through the RB particle filter (unbiased and low variance).

For simplicity we'll just check the gradient estimators at the "true" value of $\tth$, i.e., the one used to simulate the data.  In the code below, we use the term "score" for $\nabla \ell(\tth)$, which is the technical term for the gradient of the loglikelihood function. 

### Timing Comparisons

Note here that the number of particles for the basic and RB particle filters was chosen so that the gradient computations take about the same CPU time.

```{code-cell} ipython3
# exact score function
bm_score_exact = jax.jit(jax.grad(partial(bm_loglik_exact, y_meas=y_meas)))


# auto score function
@partial(jax.jit, static_argnums=(2,))
def bm_score_auto(theta, key, n_particles):
    return jax.grad(bm_loglik_basic)(theta, y_meas, key, n_particles)


# acc score function
@partial(jax.jit, static_argnums=(2,))
def bm_score_acc(theta, key, n_particles):
    out = pf.particle_filter(
        model=bm_model,
        key=key,
        y_meas=y_meas,
        theta=theta,
        n_particles=n_particles,
        score=True,
        fisher=False,
        history=False
    )
    return out["score"]


# rb score function
@partial(jax.jit, static_argnums=(2,))
def bm_score_rb(theta, key, n_particles):
    out = pf.particle_filter_rb(
        model=bm_model,
        key=key,
        y_meas=y_meas,
        theta=theta,
        n_particles=n_particles,
        score=True,
        fisher=False,
        history=False
    )
    return out["score"]


# check timings
key = jax.random.PRNGKey(0)
n_particles_basic = 2500
n_particles_rb = 100

%timeit bm_score_exact(theta_true)
%timeit bm_score_auto(theta_true, key, n_particles_basic)
%timeit bm_score_acc(theta_true, key, n_particles_basic)
%timeit bm_score_rb(theta_true, key, n_particles_rb)
```

### Accuracy Comparisons

```{code-cell} ipython3
# repeat calculation nsim times
n_sim = 100
key, *subkeys = jax.random.split(key, n_sim+1)

score_exact = bm_score_exact(theta_true)
score_auto = []
score_acc = []
score_rb = []

for i in range(n_sim):
    score_auto += [bm_score_auto(theta_true, subkeys[i], n_particles_basic)]
    score_acc += [bm_score_acc(theta_true, subkeys[i], n_particles_basic)]
    score_rb += [bm_score_rb(theta_true, subkeys[i], n_particles_rb)]
```

```{code-cell} ipython3
plot_df = (
    pd.DataFrame({
    "theta": np.tile(theta_names, n_sim),
    "auto": np.array(score_auto).ravel(),
    "acc": np.array(score_acc).ravel(),
    "rb": np.array(score_rb).ravel()
})
    .melt(id_vars=["theta"], value_vars=["auto", "acc", "rb"], var_name="method")
)

g = sns.catplot(
    data=plot_df, kind="box",
    x="method", y="value",
    col="theta",
    col_wrap=3,
    sharey=False
)
g.set_titles(col_template="{col_name}")
[g.axes[i].axhline(score_exact[i]) for i in range(theta_true.size)];
```

```{code-cell} ipython3
# same thing without auto
plot_df = (
    pd.DataFrame({
    "theta": np.tile(theta_names, n_sim),
    "acc": np.array(score_acc).ravel(),
    "rb": np.array(score_rb).ravel()
})
    .melt(id_vars=["theta"], value_vars=["acc", "rb"], var_name="method")
)

g = sns.catplot(
    data=plot_df, kind="box",
    x="method", y="value",
    col="theta",
    col_wrap=3,
    sharey=False
)
g.set_titles(col_template="{col_name}")
[g.axes[i].axhline(score_exact[i]) for i in range(theta_true.size)];
```

**Conclusions**:

- This confirms that autodiff through the particle filter is biased (in $\mu$ and $\tau$) whearas the other two filters are not.

- The RB score calculation indeed has lower variance than that of the basic particle filter.  However, the gradients for both $\sigma$ and $\tau$ appear to be somewhat more biased.

+++

## Hessian Computations

We'll do this using the same methods as for the score.

In the code below, we use the term Fisher information for $- \nabla^2 \ell(\tth)$, the technical term for the hessian of the negative loglikelihood.

### Timing Comparisons

```{code-cell} ipython3
# exact fisher information
@jax.jit
def bm_fisher_exact(theta):
    hess = jax.jacfwd(jax.jacrev(partial(bm_loglik_exact, y_meas=y_meas)))(theta)
    return -hess


# auto fisher information
@partial(jax.jit, static_argnums=(2,))
def bm_fisher_auto(theta, key, n_particles):
    return jax.jacfwd(jax.jacrev(bm_loglik_basic))(theta, y_meas, key, n_particles)


# acc fisher information
@partial(jax.jit, static_argnums=(2,))
def bm_fisher_acc(theta, key, n_particles):
    out = pf.particle_filter(
        model=bm_model,
        key=key,
        y_meas=y_meas,
        theta=theta,
        n_particles=n_particles,
        score=True,
        fisher=True,
        history=False
    )
    return out["fisher"]


# rb fisher information
@partial(jax.jit, static_argnums=(2,))
def bm_fisher_rb(theta, key, n_particles):
    out = pf.particle_filter_rb(
        model=bm_model,
        key=key,
        y_meas=y_meas,
        theta=theta,
        n_particles=n_particles,
        score=True,
        fisher=True,
        history=False
    )
    return out["fisher"]


# timing comparisons
key = jax.random.PRNGKey(0)
n_particles_basic = 2500
n_particles_rb = 100

%timeit bm_fisher_exact(theta_true)
%timeit bm_fisher_auto(theta_true, key, n_particles_basic)
%timeit bm_fisher_acc(theta_true, key, n_particles_basic)
%timeit bm_fisher_rb(theta_true, key, n_particles_rb)
```

### Accuracy Comparisons

```{code-cell} ipython3
n_sim = 100
key, *subkeys = jax.random.split(key, n_sim+1)

# repeat calculation nsim times
fisher_exact = bm_fisher_exact(theta_true)
fisher_auto = []
fisher_acc = []
fisher_rb = []

for i in range(n_sim):
    fisher_auto += [bm_fisher_auto(theta_true, subkeys[i], n_particles_basic)]
    fisher_acc += [bm_fisher_acc(theta_true, subkeys[i], n_particles_basic)]
    fisher_rb += [bm_fisher_rb(theta_true, subkeys[i], n_particles_rb)]
```

```{code-cell} ipython3
theta2_names = np.meshgrid(np.array(theta_names), np.array(theta_names))
theta2_names = np.array(
    [theta2_names[1].ravel()[i] + '_' +
     theta2_names[0].ravel()[i]
     for i in range(theta2_names[0].size)]
)

plot_df = (
    pd.DataFrame({
    "theta": np.tile(theta2_names, n_sim),
    "auto": np.array(fisher_auto).ravel(),
    "acc": np.array(fisher_acc).ravel(),
    "rb": np.array(fisher_rb).ravel()
})
    .melt(id_vars=["theta"], value_vars=["auto", "acc", "rb"], var_name="method")
)

g = sns.catplot(
    data=plot_df, kind="box",
    x="method", y="value",
    col="theta",
    col_wrap=3,
    sharey=False
)
g.set_titles(col_template="{col_name}")
[g.axes[i].axhline(fisher_exact.ravel()[i]) for i in range(theta2_names.size)];
```

```{code-cell} ipython3
# same thing without auto
plot_df = (
    pd.DataFrame({
    "theta": np.tile(theta2_names, n_sim),
    "acc": np.array(fisher_acc).ravel(),
    "rb": np.array(fisher_rb).ravel()
})
    .melt(id_vars=["theta"], value_vars=["acc", "rb"], var_name="method")
)

g = sns.catplot(
    data=plot_df, kind="box",
    x="method", y="value",
    col="theta",
    col_wrap=3,
    sharey=False
)
g.set_titles(col_template="{col_name}")
[g.axes[i].axhline(fisher_exact.ravel()[i]) for i in range(theta2_names.size)];
```

**Conclusions**:

- In this case the Rao-Blackwellized filter is the clear winner, in terms of accuracy and precision.  It does that ~10x longer than the standard filter, so a more careful evaluation with similar timings is needed.

+++

## Particle Filtering with PFJAX

Particle filter is a technique for estimating the parameters $\tth$ of a state-space model

$$
\begin{aligned}
\xx_0 & \sim \pi(\xx_0 \mid \tth) \\
\xx_t & \sim f(\xx_t \mid \xx_{t-1}, \tth) \\
\yy_t & \sim g(\yy_t \mid \xx_t, \tth).
\end{aligned}
$$

In order to do this, the user must provide:

1.  The number of particles $N$.
2.  A proposal distribution for the initial state, $q(\xx_0 \mid \yy_0, \tth)$.
3.  A proposal distribution for the subsequent states, $r(\xx_t \mid \xx_{t-1}, \yy_t, \tth)$.

A simple choice for the proposal distributions are the forward distributions

$$
\begin{aligned}
q(\xx_0 \mid \yy_0, \tth) & = \pi(\xx_0, \tth), \\
r(\xx_t \mid \xx_{t-1}, \yy_t, \tth) & = f(\xx_t \mid \xx_{t-1}, \tth).
\end{aligned}
$$

This is called a **bootstrap** particle filter {cite:p}`gordon.etal87`.

The basic particle filter (e.g., {cite:t}`johanson.doucet09`, Section 4.1) then proceeds as follows to obtain an estimate $\hat \ell(\tth \mid \yy_{0:T})$ of the marginal loglikelihood

$$
\ell(\tth \mid \yy_{0:T}) = \log \int \pi(\xx_0 \mid \yy_0) \times \prod_{t=1}^T f(\xx_{t} \mid \xx_{t-1}, \tth) \times g(\yy_t \mid \xx_t, \tth) \ud \xx_{0:T}.
$$

--- 

(sec:bpf)=
### Algorithm: Basic Particle Filter

**Inputs:** $\tth$, $\yy_{0:T}$

**Outputs:** $\hat \ell(\tth \mid \yy_{0:T})$

- $\xx_0^{(1:N)} \iid q(\xx_0 \mid \yy_0, \tth)$
    
- $w_0^{(1:N)} \gets g(\yy_0 \mid \xx_0^{(1:N)}) \cdot \pi(\xx_0^{(1:N)} \mid \tth) / q(\xx_0^{(1:N)} \mid \yy_0, \tth)$
    
-  $\hat{\Ell}_0 \gets \sum_{i=1}^N w_0^{(i)}$

- $W_0^{(1:N)} = w_0^{(1:N)}/ \hat{\Ell}_0$
    
-  For $t=1,\ldots,T$:

    -  $\tilde{\xx}_{t-1}^{(1:N)} \gets \operatorname{\texttt{resample}}(\xx_{t-1}^{(1:N)}, W_{t-1}^{(1:N)})$
    
    - $\xx_t^{(1:N)} \ind r(\xx_t \mid \tilde{\xx}_{t-1}^{(1:N)}, \yy_t, \tth)$
    
    - $w_t^{(1:N)} \gets g(\yy_t \mid \xx_t^{(1:N)} \mid \tth) \cdot f(\xx_t^{(1:N)} \mid \tilde{\xx}_{t-1}^{(1:N)}) / r(\xx_t \mid \tilde{\xx}_{t-1}^{(1:N)}, \yy_t, \tth)$
    
    - $\hat{\Ell}_t \gets \sum_{i=1}^N w_t^{(i)}$
    
    - $W_t^{(1:N)} = w_t^{(1:N)}/ \hat{\Ell}_t$

- $\hat \ell(\tth \mid \yy_{0:T}) = \sum_{t=0}^T \log \hat{\Ell}_t$

---

### Particle Resampling

In the [basic particle filter algorithm](sec:bpf), the notation $\xx_t^{(1:N)}$ stands for $\xx_t^{(1)}, \ldots, \xx_t^{(N)}$, i.e., is over the vector of $N$ particles.  Similarly, operations of the form $\xx_t^{(1:N)} \gets F(\xx_{t-1}^{(1:N)})$ are vectorized over the $N$ particles, i.e., correspond to the for-loop 

- For $i=1,\ldots,N$:

    - $\xx_t^{(i)} \gets F(\xx_{t-1}^{(i)})$
    
The $\operatorname{\texttt{resample}}()$ function takes a weighted set of particles $(\xx^{(1:N)}, W^{(1:N)})$ and attempts to convert it to an unweighted sample $\tilde{\xx}^{(1:N)}$ from the same underlying distribution.  The simplest way to do this is via sampling with replacement from $(\xx^{(1:N)}, W^{(1:N)})$.  However, this resampling function is not differentiable with respect to $\tth$ with parameter-dependent particles $(\xx^{(1:N)}(\tth), W^{(1:N)}(\tth))$.

+++

## Appendix: Exact Likelihood of the BM Model

The distribution of $p(\xx_{0:T}, \yy_{1:T} \mid y_0, \tth)$ is multivariate normal.  Thus we only need to find  $E[\yy_{1:T} \mid y_0, \tth]$ and $\var(\yy_{1:T} \mid y_0, \tth)$.

Conditioned on $x_0$ and $\tth$, the Brownian latent variables $\xx_{1:T}$ are multivariate normal with
$$
\newcommand{\cov}{\operatorname{cov}}
\begin{aligned}
%E[x_0 \mid \tth] & = y_0 & \var(x_0\\
E[x_t \mid x_0, \tth] & = x_0 + \mu t, \\
\cov(x_s, x_t \mid x_0, \tth) & = \sigma^2 \min(s, t).
\end{aligned}
$$
Conditioned on $\xx_{0:T}$ and $\tth$, the measurement variables $\yy_{1:T}$ are multivariate normal with
$$
\begin{aligned}
E[y_t \mid \xx_{0:T}, \tth] & = \xx_{1:T}, \\
\cov(y_s, y_t \mid \xx_{0:T}, \tth) & = \tau^2 \delta_{st}.
\end{aligned}
$$
Therefore, the marginal distribution of $\yy_{1:T}$ is multivariate normal with
$$
\begin{aligned}
E[y_t \mid x_0, \tth] 
& = E[E[y_t \mid \xx_{0:T}, \tth] \mid x_0, \tth] \\
& = x_0 + \mu t \\
\cov(y_s, y_t \mid x_0, \tth) 
& = \cov(E[y_s \mid \xx_{0:T}, \tth], E[y_t \mid \xx_{0:T}] \mid \tth) + E[\cov(y_s, y_t \mid \xx_{0:T}, \tth) \mid \tth] \\
& = \sigma^2 \min(s, t) + \tau^2 \delta_{st}.
\end{aligned}
$$
For the given choice of prior, we have $x_0 \mid y_0 \sim \N(y_0, \tau^2)$ for the initial observation $y_0$.  Integrating over $x_0$, the marginal distribution of $\yy_{1:T}$ is MVN with
$$
\begin{aligned}
E[y_t \mid y_0, \tth] & = y_0 + \mu t, \\
\cov(y_s, y_t \mid y_0, \tth) & = \sigma^2 \min(s, t) + \tau^2(\delta_{st} + 1).
\end{aligned}
$$

```{code-cell} ipython3
import pfjax as pf
import pfjax.models
import inspect

lines = inspect.getsource(pf.models.BMModel.state_lpdf)
print(lines)
```

```{code-cell} ipython3

```

## Check Particle Filter Approximations

Let $\hat \ell_N(\tth)$ denote a particle filter estimate of $\ell(\tth)$ with $N$ particles, and similar notations for $\widehat{\nabla \ell}_N(\tth)$ and $\widehat{\nabla^2 \ell}_N(\tth)$.  Our purpose here is to check that

$$
\begin{aligned}
\hat \ell_N(\tth) & \to \ell(\tth) \\
\widehat{\nabla \ell}_N(\tth) & \to \nabla \ell(\tth) \\
\widehat{\nabla^2 \ell}_N(\tth) & \to \nabla^2 \ell(\tth)
\end{aligned}
$$

as $N \to \infty$.  
<!-- The usual way of estimating the score and Hessian functions $\nabla \ell(\tth)$ and $\nabla^2 \ell(\tth)$ is described in e.g., Cappe et al (2005), Poyiadjis et al (2011).  Here we would like leverage the power of the JAX autodiff engine to just differentiate through the particle filter, i.e,. $\widehat{\nabla \ell}_N(\tth) = \nabla \hat{\ell}_N(\tth)$ and $\widehat{\nabla^2 \ell}_N(\tth) = \nabla^2 \hat{\ell}_N(\tth)$. -->

+++

### Loglikelihood Check

<!-- We'll proceed on the negative scale, which converts the MLE to a minimization problem for which most optimization algorithms are naturally defined.  Also, we'll convert the parameter to an unconstrained scale, $\pph = (\mu, \log \sigma, \log \tau)$.
 -->
 
We will consider two different particle filter algorithms for this problem:

1.  A standard bootstrap particle filter with multinomial resampling \cite{doucet_johansen09}.  The complexity of this particle filter is $\bO(NT)$ and the storage cost is $\bO(N)$.

2.  A Rao-Blackwellized bootstrap particle filter described by \cite{poyiadjis_etal11}.  The complexity of this algorithm is $\bO(N^2T)$ and the storage cost is $\bO(N)$.

Here we'll content ourselves with a visual assessment using projection plots about the true parameter value.
 
**Warning:** The code below occasionally makes use of global variables, thus violating the functional programming paradigm of JAX.  Changing these global variables after jitting will lead to incorrect results.

```{code-cell} ipython3
def to_phi(theta):
    """
    Helper function to convert theta to phi.
    """
    return jnp.array([theta[0], jnp.log(theta[1]), jnp.log(theta[2])])

def to_theta(phi):
    """
    Helper function to convert phi to theta.
    """
    return jnp.array([phi[0], jnp.exp(phi[1]), jnp.exp(phi[2])])

def prop_lpdf(self, x_curr, x_prev, y_curr, theta):
    """
    Add proposal log-pdf to bm_model.
    """
    return self.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)

def bm_loglik_exact(theta, y_meas):
    """
    Exact loglikelihood of the BM model.
    """
    return bm_model.loglik_exact(y_meas=y_meas, theta=theta)

def bm_loglik_stan(theta, y_meas, key, n_particles):
    """
    Standard particle filter approximation of the loglikelihood.
    """
    pf_out = pfex.particle_filter(
        model=bm_model,
        key=key,
        y_meas=y_meas,
        theta=theta,
        n_particles=n_particles,
        history=False
    )
    return pf_out["loglik"]

def bm_loglik_rb(theta, y_meas, key, n_particles):
    """
    Rao-Blackwellized particle filter approximation of the negative loglikelihood.
    """
    pf_out = pfex.particle_filter_rb(
        model=bm_model,
        key=key,
        y_meas=y_meas,
        theta=theta,
        n_particles=n_particles,
        history=False
    )
    return pf_out["loglik"]
```

```{code-cell} ipython3
def proj_data(fun, x, x_lims, x_names, n_pts=100):
    """
    Wrapper for `projplot.projxvals()` and `projplot.projdata()`.

    Won't need this for upcoming interface of projplot.
    """
    xvals = pjp.projxvals(x, x_lims, n_pts)
    return pjp.projdata(fun, xvals, x_names, is_vectorized=False)


# plot exact likelihood
theta_lims = jnp.array([[4., 6.], [.01, .4], [.5, 2]])  # for n_obs = 100
# theta_lims = jnp.array([[4., 5.5], [.01, .4], [.8, 1.4]]) # for n_obs = 10
# phi_true = to_phi(theta_true)
# phi_lims = to_phi(theta_lims)

theta_names = ["mu", "sigma", "tau"]
n_pts = 100

# calculate projection plot
df_exact = pjp.proj_plot(
    fun=jax.jit(partial(bm_loglik_exact, y_meas=y_meas)),
    x_opt=theta_true, 
    x_lims=theta_lims, 
    x_names=theta_names, 
    n_pts=n_pts
)
```

```{code-cell} ipython3
# standard particle filter

n_particles_stan = 2500

key, subkey = random.split(key)
bm_ll_stan = jax.jit(partial(bm_loglik_stan, y_meas=y_meas,
                             n_particles=n_particles_stan, key=subkey))

%timeit bm_ll_stan(theta_true)

df_stan = proj_data(bm_ll_stan,
                    theta_true, theta_lims, theta_names)
```

```{code-cell} ipython3
# rao-blackwellized particle filter

n_particles_rb = 100

key, subkey = random.split(key)
bm_ll_rb = jax.jit(partial(bm_loglik_rb, y_meas=y_meas,
                             n_particles=n_particles_rb, key=subkey))

%timeit bm_ll_rb(theta_true)

df_rb = proj_data(bm_ll_rb,
                    theta_true, theta_lims, theta_names)
```

```{code-cell} ipython3
plot_df = pd.concat([df_exact, df_stan, df_rb], ignore_index=True)
plot_df["method"] = np.repeat(["exact", "pf_stan", "pf_rb"], len(df_exact["theta"]))
rp = sns.relplot(
    data=plot_df, kind="line",
    x="x", y="y", 
    hue="method",
    col="theta",
    col_wrap = 3,
    facet_kws=dict(sharex=False, sharey=False)
)
rp.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
rp.fig.suptitle('Projection Plots using Closed-Form Latents');
```

**Conclusions:** 

- We used 1000 particles for the standard filter but only 100 particles for the Rao-Blackwellized filter.  The latter takes longer to compute but supposedly has lower variance.  In this particular case it does not seem to be the case, i.e., the Rao-Blackwellized filter takes longer and appears to be more variable.

- Both particle filters reasonably approximate $\ell(\tth)$ when $\mu$ is at its true value.  They don't do so well when $\mu$ is far from its true value.  This may have to do with particle degeneracy in that case.
