# pjfax 0.0.0.9001

## Breaking Changes

### Changes to Function Names, Arguments, Outputs

- [ ] `particle_resample()`, `particle_resample_mvn()`, etc. -> `resample_multinomial()`, `resample_mvn()`, etc.

- [ ] `logw` -> `lwgt`.

- [ ] `particle_resampler` argument to `particle_filter()` -> `resampler`.

- [x] In `particle_resample_mvn{_for}()`, outputs `x_particles_mu`, `x_particles_cov` -> `mean`, `cov`.

### Other

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

