# Changelog

## [0.0.3] -- (In Progress)

### Breaking Changes

- Dropped support for Python 3.9.

- Minimal versions of dependencies now specified in `pyproject.toml`.

- Project website now at <https://mlysy.github.io/pfjax/>.

### New Features

- Added full PyTree support to all pfjax functions.  Not full tested yet though...

- Added a new version of `SDEModel` in `pfjax.experimental`.  Fixes a few bugs and better implementation of bridge proposal.

- Added `ContinuousTimeModel` to `pfjax.experimental`, from which `SDEModel` is derived.

- Reimplemented `BaseModel` in `pfjax.experimental.BaseModel`.  All of the experimental models will eventually supercede their non-experimental counterparts.

- Implemented a jittable version of Sinkhorn algorithm in `pfjax.resamples.resample_ot()`.

### Bug Fixes

- Requiring `pillow>=12.1.1` to address security vulnerabilities.

- Created tree indexing utilities and corrected broadcasting logic in `_tree_mean`.

- Standardized JAX syntax by switching from `jnp.alltrue()` to `jnp.all()`.

- Fixed duplicated use of PRNG key in `sde.SDEModel`.

- Fixed test failures in `test_particle_filter_rb`.

- Replaced `import *` with explicit imports across the package.

<!-- - Updated import structure to use `__all__` in `__init__.py`. -->

### Maintenance

#### Core

- Switched to `uv` for project management.

#### Documentation
 
- Switched to Quarto + MkDocs-Material for documentation.

- Created a `gh-pages` deployment workflow with separated execution steps.


#### Testing

- Major refactoring of tests and extensive documentation added to these.

- Switched to `pytest` for unit testing.

- Created GitHub Actions for minimum-dependency testing on Python 3.11-3.14.

- Created `StoVolModel` in `pfjax.experimental.models` for testing `pfjax.experimental` module.

---

## [0.0.2] -- 2024-11-12

### Breaking Changes

- Dropped support for Python 3.6.

### New Features

- Introduced the `BaseModel` class as the foundation for all library models.

- Switched to SVD decomposition in `pfjax.resamplers.mvn_resampler()` to allow for degenerate variances.

- Added optimal transport resampling in `pfjax.resamplers.resample_ot()`.

- Created website for documentation at <https://pfjax.readthedocs.io/>.

### Bug Fixes

- Formally tested `particle_filter` and `particle_filter_rb` implementations.

- Fixed unit tests and broadcasting errors in `_tree_mean`.

<!-- - Added Ito transform to the Lotka-Volterra model. -->

### Maintenance

#### Refactoring

- Standardized `*_for()` methods as private functions in `pfjax/test`.

- Factored `_bridge_mv()` out of the SDEModel bridge step.

- Using `__all__` in `__init__.py` to properly flatten imports.

- Decoupled adaptation logic from the stepping process in the `AdaptiveMWG` class.

#### Documentation

- Created draft of `qm_gle` tutorial.

- Created draft of `svj` tutorial.

- Created draft of `lotvol` tutorial.

#### Testing

- Added Python 3.12 support to the test matrix.

- Updated `tox` workflow to run checks on Pull Requests.



---

## [0.0.1] -- 2022-03-30

### Breaking Changes

#### Changes to Function Names, Arguments, Outputs

- [x] `particle_filter()` has different outputs as described in the docstrip.

- [x] `particle_resample()`, `particle_resample_mvn()`, etc. -> `resample_multinomial()`, `resample_mvn()`, etc.

- [x] ~~`logw` -> `lwgt`.~~

	**Update:** Decided against this.  Even though `lwgt` is slightly more informative, its harder to pronounce and therefore more cumbersome.

- [x] `_lweight_to_prob()` -> `logw_to_prob()`.

- [x] `particle_sampler` argument to `particle_filter()` -> `resampler`.

- [x] In `particle_resample_mvn{_for}()`, outputs `x_particles_mu`, `x_particles_cov` -> `mean`, `cov`.

- [x] `full_loglik{_for}()` -> `loglik_full{_for}()`. 

- [x] `SDEModel.bridge_prop()` -> `SDEModel.bridge_step()`.

#### Other

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

### Backward-Compatible Changes

- [x] Added `score`, `fisher`, and `history` flags to `particle_filter()` for computing derivatives via the (unbiased) accumulator method, and for keeping or discarding the entire particle history (discard by default).

- [x] Added `particle_filter_rb()` which has quadratic complexity but computes the score an hessian much more precisely than the method in `particle_filter()`.

- [x] Added `particle_filter.pf_resampler_ot()` for resampling via (regularized) optimal transport.

- [x] Created an `experimental` module for placing code that's useful but that general users should not rely on: anything in the `experimental` module is subject to change/disappear without any notice!

- [x] Added `mvn_bridge` module which can e.g., be used for SDE bridge proposals.

- [x] Moved `examples` folder to `docs/notebooks/internal`.

