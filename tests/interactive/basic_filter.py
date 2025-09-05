import jax
import jax.numpy as jnp
from pfjax.particle_resamplers import resample_multinomial


class BasicFilter(object):
    def __init__(self):
        """
        This is where the private members could be defined.

        For now there aren't any.
        """
        pass

    def __call__(
        self,
        model,
        key,
        y_meas,
        theta,
        n_particles,
        resampler=resample_multinomial,
        score=False,
        fisher=False,
        history=False,
    ):
        """
        This should be essentially equivalent to previous calls to `particle_filter()`.

        Should some of the arguments be removed in favor of passing them in via constructor?

        No.  Easiest to just do them all here.
        """

        # lax.scan stepping function
        def filter_step(carry, y_curr):
            # 1. resample particles from previous time point
            key, subkey = random.split(carry["key"])
            logw_aux = jax.vmap(self.aux_lpdf, in_axes(0, None, None))(
                carry["x_particles"],
                y_curr,
                theta,
            )
            resample_out = resampler(
                key=subkey,
                x_particles_prev=carry["x_particles"],
                logw=carry["logw"] + logw_aux,
            )
            # 2. sample particles for current time point
            key, *subkeys = random.split(key, num=n_particles + 1)
            x_particles, logw = jax.vmap(model.pf_step, in_axes=(0, 0, None, None))(
                jnp.array(subkeys),
                resample_out["x_particles"],
                y_curr,
                theta,
            )
            logw_aux = jax.vmap(self.aux_lpdf, in_axes(0, None, None))(
                resample_out["x_particles"],
                y_curr,
                theta,
            )
            logw = logw - logw_aux
            # 3. accumulate quantities for score and/or fisher information
