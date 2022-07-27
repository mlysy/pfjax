import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
import pfjax as pf
import pfjax.mcmc as mcmc
from pfjax import sde as sde
from functools import partial


"""
General model for stoichiometry models casted into SDEs.
"""

class StoichModel(sde.SDEModel):

    def __init__(self, dt, n_res, InMatrix_Full, OutMatrix_Full, mask=None, epsilon=1e-6, bootstrap=True):

        super().__init__(dt, n_res, diff_diag=False) # Inherits SDEModel class
        self._bootstrap = bootstrap
        
        # Full stoichiometry matrix
        self._InMatrix_Full = InMatrix_Full
        self._OutMatrix_Full = OutMatrix_Full
        self._StoichMatrix_Full =  self._OutMatrix_Full - self._InMatrix_Full
        self._n_X_full, self._n_Rxn_full = self._StoichMatrix_Full.shape
        
        # Linearly independent stoichiometry matrix
        q,r = jnp.linalg.qr(jnp.transpose(self._StoichMatrix_Full))
        self._mask = jnp.array(jnp.abs(jnp.diag(r))>=epsilon) if mask is None else mask
        
        # Row-reduced stoichiometry matrix
        self._InMatrix = self._InMatrix_Full[self._mask,:]
        self._OutMatrix = self._OutMatrix_Full[self._mask,:]
        self._StoichMatrix = self._StoichMatrix_Full[self._mask,:]
        self._n_X, self._n_Rxn = self._StoichMatrix.shape
        
        self._n_state = (self._n_res, self._n_X)
        
        # The link matrix to restore dependent species
        self._L = jnp.matmul(self._StoichMatrix_Full, jnp.linalg.pinv(self._StoichMatrix))
        mask_matrix = jnp.transpose(jnp.broadcast_to(self._mask, (self._n_X, self._n_X_full))) # mask broadcasted to matrix
        self._L0 = jnp.where(jnp.invert(mask_matrix), self._L ,jnp.zeros((self._n_X_full,self._n_X))) 
        
    def _Hazard(self, x, param):
        
        # Propensity of i-th reaction determined by reaction type
        def h(x, param_i, i):

            Rxn = self._InMatrix_Full[:,i]

            n_mols = sum(Rxn)
            n_type = jnp.count_nonzero(Rxn)
            index = jnp.nonzero(jnp.array(Rxn),size=2)

            if n_mols == 0:
                ans = param_i
            elif n_mols == 1:
                ans = param_i * x[index[0][0]]
            elif n_mols == 2 and n_type == 1:
                ans = param_i * x[index[0][0]] * (x[index[0][0]] - 1) / 2
            elif n_mols == 2 and n_type == 2:
                ans = param_i * jnp.prod(x[index[0]])
            else:
                # WORK NEEDED
                print('Not supported reaction')
                # Throw some error
            return ans
        
        Hazard = jnp.array([h(x, param[i], i) for i in range(self._n_Rxn)])
        
        return Hazard
    
    # Restore population of dependent species, given current (independent) and initial (full) population
    def _x_full(self, x, x_init):
        
        T_tilde = jnp.where(jnp.invert(self._mask), x_init, jnp.zeros(self._n_X_full)) - self._L0 @ x_init[self._mask]
        x_full = self._L @ x + T_tilde
        
        return x_full
    
    # Drift on the regular scale
    def _drift(self, x, theta):
        x_full = self._x_full(x, theta[(self._n_Rxn + self._n_X):])
        Hazard = self._Hazard(x_full, theta[:self._n_Rxn])
        mu = self._StoichMatrix @ Hazard
        
        return mu
    
    # Diffusion on the regular scale
    def _diff(self, x, theta):
        x_full = self._x_full(x, theta[(self._n_Rxn + self._n_X):])
        Hazard = self._Hazard(x_full, theta[:self._n_Rxn])
        Sigma = self._StoichMatrix @ jnp.diag(Hazard) @ jnp.transpose(self._StoichMatrix)

        return Sigma
    
    # Drift on the log scale
    def drift(self, x, theta):
        x = jnp.exp(x)
        mu = self._drift(x, theta)
        Sigma = self._diff(x, theta)

        f_p = 1/x
        f_pp = -1/x/x

        mu_trans = f_p * mu + 0.5 * f_pp * jnp.diag(Sigma)
        return mu_trans
    
    # Diffusion on the log scale
    def diff(self, x, theta):
        x = jnp.exp(x)
        Sigma = self._diff(x, theta)

        f_p = 1/x
        Sigma_trans = jnp.outer(f_p, f_p) * Sigma

        return Sigma_trans
    
    def meas_lpdf(self, y_curr, x_curr, theta):
        """
        Log-density of `p(y_curr | x_curr, theta)`.
        Args:
            y_curr: Measurement variable at current time `t`.
            x_curr: State variable at current time `t`.
            theta: Parameter value.
        Returns
            The log-density of `p(y_curr | x_curr, theta)`.
        """
        tau = theta[self._n_Rxn:(self._n_Rxn+self._n_X)]
        return jnp.sum(
            jsp.stats.norm.logpdf(y_curr, loc=jnp.exp(x_curr[-1]), scale=tau)
        )
        
    def meas_sample(self, key, x_curr, theta):
        """
        Sample from `p(y_curr | x_curr, theta)`.
        Args:
            key: PRNG key.
            x_curr: State variable at current time `t`.
            theta: Parameter value.
        Returns:
            Sample of the measurement variable at current time `t`: `y_curr ~ p(y_curr | x_curr, theta)`.
        """
        tau = theta[self._n_Rxn:(self._n_Rxn+self._n_X)]
        return jnp.exp(x_curr[-1]) + tau * random.normal(key, (self._n_state[1],))
    
    def pf_init(self, key, y_init, theta):
        """
        Particle filter calculation for `x_init`.
        Samples from an importance sampling proposal distribution
        ```
        x_init ~ q(x_init) = q(x_init | y_init, theta)
        ```
        and calculates the log weight
        ```
        logw = log p(y_init | x_init, theta) + log p(x_init | theta) - log q(x_init)
        ```
        **FIXME:** Explain what the proposal is and why it gives `logw = 0`.
        In fact, if you think about it hard enough then it's not actually a perfect proposal...
        Args:
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.
            key: PRNG key.
        Returns:
            - x_init: A sample from the proposal distribution for `x_init`.
            - logw: The log-weight of `x_init`.
        """
        tau = theta[self._n_Rxn:(self._n_Rxn+self._n_X)]

        key, subkey = random.split(key)
        x_init = jnp.log(y_init + tau * random.truncated_normal(
            subkey,
            lower=-y_init/tau,
            upper=jnp.inf,
            shape=(self._n_state[1],)
        ))
        logw = jnp.sum(jsp.stats.norm.logcdf(y_init/tau))
        
        return \
            jnp.append(jnp.zeros((self._n_res-1,) + x_init.shape),
                       jnp.expand_dims(x_init, axis=0), axis=0), \
            logw
    
    def pf_step(self, key, x_prev, y_curr, theta):
        """
        Choose between bootstrap filter and bridge proposal.
        Args:
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.
            key: PRNG key.
        Returns:
            - x_curr: Sample of the state variable at current time `t`: `x_curr ~ q(x_curr)`.
            - logw: The log-weight of `x_curr`.
        """
        if self._bootstrap:
            x_curr, logw = super().pf_step(key, x_prev, y_curr, theta)
        else:
            omega = (theta[self._n_Rxn:(self._n_Rxn+self._n_X)] / y_curr)**2
            x_curr, logw = self.bridge_prop(
                key, x_prev, y_curr, theta, 
                jnp.log(y_curr), jnp.eye(4), jnp.diag(omega)
            )
        return x_curr, logw
