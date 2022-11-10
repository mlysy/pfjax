# pjfax 0.0.1

## Breaking Changes

### Changes to Function Names, Arguments, Outputs

- [x] `particle_filter()` has different outputs as described in the docstrip.

- [x] `particle_resample()`, `particle_resample_mvn()`, etc. -> `resample_multinomial()`, `resample_mvn()`, etc.

- [ ] `logw` -> `lwgt`.

- [x] `_lweight_to_prob()` -> `lwgt_to_prob()`.

- [x] `particle_sampler` argument to `particle_filter()` -> `resampler`.

- [x] In `particle_resample_mvn{_for}()`, outputs `x_particles_mu`, `x_particles_cov` -> `mean`, `cov`.

- [x] `full_loglik{_for}()` -> `loglik_full{_for}()`. 

### Other

- [x] Moved testing functions to the following locations:

	- Things only used for formal unit testing are in `tests/utils.py`.
	
	- Things used both for formal and informal (i.e., interactive) unit testing are in `src/pfjax/test`.  This is so that the containing modules can be imported from `tests/interactive`, which is not a submodule of **pfjax**.

- [x] Models won't be loaded by default, instead will be in the `models` module, i.e.,

	```python
	import pfjax as pf
	import pf.models
	```
	
- [x] Similarly MCMC algorithms will be in the `mcmc` module. 

	Also, interface via `AdaptiveMWG` class is totally different than the previous functional approach.

- [x] Removed `proj_data()`.  This is now contained in a separate package `kanikadchopra/projplot`.

## Backward-Compatible Changes

- [x] Added `score`, `fisher`, and `history` flags to `particle_filter()` for computing derivatives via the (unbiased) accumulator method, and for keeping or discarding the entire particle history (discard by default).

- [x] Added `particle_filter_rb()` which has quadratic complexity but computes the score an hessian much more precisely than the method in `particle_filter()`.

- [x] Added `particle_filter.pf_resampler_ot()` for resampling via (regularized) optimal transport.

- [x] Created an `experimental` module for placing code that's useful but that general users should not rely on: anything in the `experimental` module is subject to change/disappear without any notice!

- [x] Added `mvn_bridge` module which can e.g., be used for SDE bridge proposals.

- [x] Moved `examples` folder to `docs/notebooks/internal`.

