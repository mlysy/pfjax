import jax
import jax.numpy as jnp
import pfjax.utils as utils
from jax.scipy.special import logsumexp


def resample_multinomial(key, x_particles, logw):
    r"""
    Multinomial particle resampler.

    Resamples particles with replacement proportional to the weights.

    Args:
        key: PRNG key.
        x_particles: A PyTree with leading dimension `n_particles` consisting
            of the particles.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.

    Returns:
        A dictionary with elements:
            - `x_particles`: A PyTree leading dimension `n_particles` consisting
              of resampled particles.  These are sampled with replacement
              from the input `x_particles` with probability vector
              `prob = exp(logw) / sum(exp(logw))`.

            - `logw`: A vector of resampled particle weights.  TODO...

            - `ancestors`: Vector of `n_particles` integers TODO...
    """
    prob = utils.logw_to_prob(logw)
    n_particles = logw.size
    ancestors = jax.random.choice(
        key,
        a=jnp.arange(n_particles),
        shape=(n_particles,),
        p=jax.lax.stop_gradient(prob),
    )
    # note: must normalize first
    logw_resamp = jnp.log(prob)[ancestors]
    logw_resamp = (
        jnp.log(n_particles) + logw_resamp - jax.lax.stop_gradient(logw_resamp)
    )
    return {
        "x_particles": utils.tree_subset(x_particles, index=ancestors),
        "logw": logw_resamp,
        "ancestors": ancestors,
    }


class BasicFilter(object):
    def __init__(self, model):
        """
        This is where the private members could be defined.

        The only one is model.

        Note that we're also skipping score and fisher, as we'll be using
        AD for these.
        """
        self._model = model
        self._has_pf_aux = callable(getattr(self._model, "pf_aux", None))

    def initialize(self, key, y_init, theta):
        r"""
        Draw the particle set for the initial timepoint.

        `jax.vmap()` operates over `key`.
        """
        return jax.vmap(
            fun=self._model.pf_init,
            in_axes=(0, None, None),
        )(key, y_init, theta)

    def propagate(self, key, x_prev, y_curr, theta):
        r"""
        Draw the particle set for subsequent timepoints.

        `jax.vmap()` operates over `key` and `x_prev`.
        """
        return jax.vmap(
            fun=self._model.pf_step,
            in_axes=(0, 0, None, None),
        )(key, x_prev, y_curr, theta)

    def pf_aux(self, x_prev, y_curr, theta):
        r"""
        Compute the auxiliary weight term.

        `jax.vmap()` operates over `x_prev`.
        If `model.pf_aux()` is missing then return an array of zeros.
        """
        if self._has_pf_aux:
            logw_aux = jax.vmap(
                fun=self._model.pf_aux,
                in_axes=(0, None, None),
            )(x_prev, y_curr, theta)
        else:
            n_particles = jax.tree.leaves(x_prev)[0].shape[0]
            logw_aux = jnp.zeros(shape=(n_particles,))
        return logw_aux

    def __call__(
        self,
        key,
        y_meas,
        theta,
        n_particles,
        resampler=resample_multinomial,
        history=False,
    ):
        """
        Compute the particle filter loglikelihood estimate.

        **Notes:**

        - `key` for next observation generated "online" during `lax.scan()`,
          instead of pre-generated at the beginning.  This is in anticipation
          of online filtering; see below.

        - Give more flexibilty to resampler, i.e., allow weighted particle
          output.

        - Could have the option of adding new measurements as we go along.
          So for example, could have an argument `init=None`, which if not
          `None` is the carry from `lax.scan()`.  Should then also return
          the carry as an output.

        Args:
            model: Object specifying the state-space model having the following
                methods:

                - `pf_init : (key, y_init, theta) -> (x_particles, logw)`:
                  For sampling and calculating log-weights for the initial
                  latent variable.

                - `pf_step : (key, x_prev, y_curr, theta) -> (x_particles, logw)`:
                  For sampling and calculating log-weights for each subsequent
                  latent variable.

                - `pf_aux : (x_prev, y_curr, theta) -> logw`:
                  Optional method providing look-forward log-weights of the
                  auxillary particle filter.

            key: PRNG key.

            y_meas: JAX array with leading dimension `n_obs` containing the
                measurement variables `y_meas = (y_0, ..., y_T)`,
                where `T = n_obs-1`.

            theta: Parameter value.

            n_particles: Number of particles.

            resampler: Function used at step `t` to obtain sample of particles
                from `p(x_{t} | y_{0:t}, theta)` out of a sample of particles
                from `p(x_{t-1} | y_{0:t-1}, theta)`.
                The argument signature is `resampler(x_particles, logw, key)`,
                and the return value is a dictionary with mandatory element
                `x_particles` and optional elements that get carried to the
                next step `t+1` via `lax.scan()`.

            history: Whether to output the history of the filter or only the
                last step.

        Returns:
            A tuple of which the first element is the loglikelihood estimate
            and the second is a dictionary with the following elements:

            - **x_particles** - JAX array containing the state variable
              particles at the last time point (leading dimension `n_particles`)
              or at all time points (leading dimensions `(n_obs, n_particles)`
              if `history=True`.

            - **logw** - JAX array containing unnormalized log weights at the
              last time point (dimensions `n_particles`) or at all time points
              (dimensions (n_obs, n_particles)`) if `history=True`.

            - **resample_out** - If `history=True`, a dictionary of additional
              outputs from `resampler` function.  The leading dimension of each
              element of `resample_out` has leading dimension `n_obs-1`,
              since these additional outputs do not apply to the first
              time point `t=0`.
        """
        n_obs = jax.tree.leaves(y_meas)[0].shape[0]

        # lax.scan: initial value
        key, *subkeys = jax.random.split(key, num=n_particles + 1)
        x_particles, logw = self.initialize(
            key=jnp.array(subkeys),
            y_init=utils.tree_subset(y_meas, 0),
            theta=theta,
        )
        filter_init = {
            "x_particles": x_particles,
            "logw": logw,
            "loglik": logsumexp(logw) - jnp.log(n_particles),
            "key": key,
        }

        # lax.scan stepping function
        def filter_step(carry, y_curr):
            # 1. resample particles from previous time point
            logw_prev = carry["logw"]
            x_particles_prev = carry["x_particles"]
            # upweight with auxiliary pf
            logw_aux = self.pf_aux(
                x_prev=x_particles_prev,
                y_curr=y_curr,
                theta=theta,
            )
            key, subkey = jax.random.split(carry["key"])
            resample_out = resampler(
                key=subkey,
                x_particles=x_particles_prev,
                logw=logw_prev + logw_aux,
            )
            x_particles_resamp = resample_out["x_particles"]
            # FIXME: allow resampler to output weighted particles.
            # # default to unweighted now.
            # logw_resamp = jnp.zeros(shape=(n_particles,))
            # # FIXME: this should be done inside resample_multinomial.
            # logw_hist = logw_prev[resample_out["ancestors"]]
            # logw_resamp = logw_resamp + logw_hist - jax.lax.stop_gradient(logw_hist)
            logw_resamp = resample_out["logw"]
            # 2. sample particles for current time point
            key, *subkeys = jax.random.split(key, num=n_particles + 1)
            x_particles_curr, logw_curr = self.propagate(
                key=jnp.array(subkeys),
                x_prev=x_particles_resamp,
                y_curr=y_curr,
                theta=theta,
            )
            # FIXME: replace proposal with stop-gradient version
            # x_particles_curr = jax.lax.stop_gradient(x_particles_curr)
            # logw_prop = jax.vmap(
            #     fun=self._model.step_lpdf,
            #     in_axes=(0, 0, None, None),
            # )(x_particles_curr, x_particles_resamp, y_curr, theta)
            # logw_curr = logw_curr + logw_prop - jax.lax.stop_gradient(logw_prop)
            # 3. update logw
            # downweight with auxiliary pf
            logw_aux_resamp = self.pf_aux(
                x_prev=x_particles_resamp,
                y_curr=y_curr,
                theta=theta,
            )
            logw_curr = logw_curr + logw_resamp - logw_aux_resamp
            # 4. compute marginal likelihood term
            logw_marg = logsumexp(logw_curr)
            logw_marg = logw_marg - logsumexp(logw_resamp)
            logw_marg = logw_marg + logsumexp(logw_aux + logw_prev)
            logw_marg = logw_marg - logsumexp(logw_prev)
            # 5. update lax.scan carry and stack
            res_carry = {
                "x_particles": x_particles_curr,
                "logw": logw_curr,
                "key": key,
                "loglik": carry["loglik"] + logw_marg,
            }
            if history:
                # mandatory elements
                res_stack = {k: res_carry[k] for k in ["x_particles", "logw"]}
                if set(["x_particles"]) < resample_out.keys():
                    # other elements of resample_out if they exist
                    res_stack["resample_out"] = utils.rm_keys(
                        x=resample_out, keys="x_particles"
                    )
            else:
                res_stack = None

            return res_carry, res_stack

        # lax.scan: execute
        last, full = jax.lax.scan(
            f=filter_step,
            init=filter_init,
            xs=utils.tree_subset(y_meas, jnp.arange(1, n_obs)),
        )

        # format output
        loglik = last["loglik"]
        if history:
            # append initial values of x_particles and logw
            full["x_particles"] = utils.tree_append_first(
                tree=full["x_particles"], first=filter_init["x_particles"]
            )
            full["logw"] = utils.tree_append_first(
                tree=full["logw"], first=filter_init["logw"]
            )
        else:
            full = last.copy()
            # del full["loglik"]  # hold off on this for now
        return loglik, full
