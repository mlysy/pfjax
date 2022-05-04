# pjfax 0.0.1

## Breaking Changes

### Changes to Function Names, Arguments, Outputs

- [x] `particle_resample()`, `particle_resample_mvn()`, etc. -> `resample_multinomial()`, `resample_mvn()`, etc.

- [ ] `logw` -> `lwgt`.

- [x] `_lweight_to_prob()` -> `lwgt_to_prob()`.

- [x] `particle_sampler` argument to `particle_filter()` -> `resampler`.

- [x] In `particle_resample_mvn{_for}()`, outputs `x_particles_mu`, `x_particles_cov` -> `mean`, `cov`.

- [x] `full_loglik{_for}()` -> `loglik_full{_for}()`. 

### Other

- [x] Moved testing functions e.g., `*_for()` to `tests/utils.py`.

- [x] Models won't be loaded by default, instead will be in the `models` module, i.e.,

	```python
	import pfjax as pf
	import pf.models
	```
	
- [x] Similarly MCMC algorithms will be in the `mcmc` module. 

- [x] Removed `proj_data()`.  This is now contained in a separate package `kanikadchopra/projplot`.

## Backward-Compatible Changes

- [x] Added `particle_filter.pf_resampler_ot()` for resampling via (regularized) optimal transport.

- [x] Created an `experimental` module for placing code that's useful but that general users should not rely on: anything in the `experimental` module is subject to change/disappear without any notice!

	Currently this module contains `stoch_opt.py` and the `neg_loglik_*()` functions.

- [x] Added `mvn_bridge` module which can e.g., be used for SDE bridge proposals.

