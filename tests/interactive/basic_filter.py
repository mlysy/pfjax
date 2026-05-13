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
    return {
    "x_particles": utils.tree_subset(x_particles, index=ancestors),
    "ancestors": ancestors,
    "logw": -jnp.log(n_particles) * jnp.ones((n_particles,)),
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
        logw = logw - jnp.log(n_particles)  # matches pseudocode: logw = logw - log(N)
        filter_init = {
            "x_particles": x_particles,
            "logw": logw,
            "loglik": 0.0,
            "key": key,
        }

        # lax.scan stepping function
        def filter_step(carry, y_curr):
            logw = carry["logw"]
            x_particles = carry["x_particles"]

            # upweight by aux pf (== 0 if model has no pf_aux)
            logw_aux = self.pf_aux(
                x_prev=x_particles,
                y_curr=y_curr,
                theta=theta,
            )
            logw = logw + logw_aux

            # loglik contribution for this step. The stop_gradient is needed for
            # reinforce (prevents pathwise gradients from double-counting with logw_ad).
            loglik_inc = jax.lax.stop_gradient(logsumexp(logw))

            # resample
            key, subkey = jax.random.split(carry["key"])
            resample_out = resampler(
                key=subkey,
                x_particles=x_particles,
                logw=logw,
            )
            ancestors = resample_out["ancestors"]

            # Needed for reinforce: must be computed before logw is overwritten,
            # since it has to reference the weights the resampler actually saw.
            logw_ad = logw[ancestors] - jax.lax.stop_gradient(logw[ancestors])

            x_particles = resample_out["x_particles"]
            logw_prev = resample_out["logw"]  # uniform weights after resampling

            # Downweight by aux pf at the resampled particles
            logw_aux_resamp = self.pf_aux(
                x_prev=x_particles,
                y_curr=y_curr,
                theta=theta,
            )

            # Propagate. Overwrites x_particles and logw with the new step's outputs,
            # exactly like the pseudocode's `x_particles, logw = propagate(...)`.
            key, *subkeys = jax.random.split(key, num=n_particles + 1)
            x_particles, logw = self.propagate(
                key=jnp.array(subkeys),
                x_prev=x_particles,
                y_curr=y_curr,
                theta=theta,
            )

            """
            logw_aux_resamp = self.pf_aux(
                x_prev=x_particles_resamp,
                y_curr=y_curr,
                theta=theta,
            )
            logw_curr = logw_curr + logw_resamp - logw_aux_resamp

            logw_marg = logsumexp(logw_curr)
            logw_marg = logw_marg - logsumexp(logw_resamp)
            logw_marg = logw_marg + logsumexp(logw_aux + logw_prev)
            logw_marg = logw_marg - logsumexp(logw_prev)
            """

            # Combine. Pseudocode is logw = logw + logw_prev; the - logw_aux_resamp
            # is the aux-pf downweight; the + logw_ad is needed for reinforce.
            logw = logw + logw_prev - logw_aux_resamp + logw_ad

            res_carry = {
                "x_particles": x_particles,
                "logw": logw,
                "key": key,
                "loglik": carry["loglik"] + loglik_inc,
            }
            if history:
                res_stack = {k: res_carry[k] for k in ["x_particles", "logw"]}
                if set(["x_particles"]) < resample_out.keys():
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
        loglik = last["loglik"] + logsumexp(last["logw"])
        if history:
            full["x_particles"] = utils.tree_append_first(
                tree=full["x_particles"], first=filter_init["x_particles"]
            )
            full["logw"] = utils.tree_append_first(
                tree=full["logw"], first=filter_init["logw"]
            )
        else:
            full = last.copy()
        return loglik, full