---
title: "Test Layout"
---

# Generic Method Tests

## `simulate()`

- `test_for`: Test that for-loop and vmap/lax.scan are identical.
- `test_jit`: Test that jitted and unjitted versions are the same.  This includes taking gradients.

## `loglik_full()`

- `test_for`
- `test_jit`

## `particle_filter()`

- `test_for`
- `test_jit`: Not implemented...todo?
- `test_deriv`: Test gradient and hessian computations, which aren't just obtained via autodiff.

## `particle_filter_rb()`

- `test_for`
- `test_jit`: Not implemented...todo?
- `test_deriv`
- `test_history`: Test whether or not saving full particle history gives the same result.

## `particle_smooth()`

- `test_for`
- `test_jit`

## Improvements

- Test with PyTree inputs.

# Resampler Tests

## `resample_mvn()`

- `test_for`
- `test_jit`
- `test_shape`: Test that resampler does the right thing for arrays of various shapes.  PyTrees are supported but untested.

## `resample_ot()`

- `test_sinkhorn`: Test that custom and **ott** package implementation of Sinkhorn algorithm give the same result.
- `test_jit`: Test various ways of jitting arguments to `resample_ot()` which are passed on to **ott**.

## Improvements

- Add tests for `resample_multinomial()`.
- Test that the resamplers works for PyTrees.
- Test that `resampler_mvn()` works for degenerate MVNs.


# SDEModel Tests

## Inheritance Test

- `test_simulate_models`: Test that hard-coded and derived implementation of given SDE model(s) give the same results under `simulate()`.
- `test_loglik_full_models`: Same thing, but with `loglik_full()`.

## Jit Test

- `test_simulate_jit`: Run the jit/grad tests of `simulate()` method on the given model.
- `test_loglik_full_jit`: Same but with `loglik_full()`.

## For-Loop Test

- `test_sde_state_sample_for`: Test that `state_sample()` for SDEs using for-loop and lax.scan are the same.  This test currently only works for `LotVolModel`, for which the for-loop version has been hard-coded into the class.
- `test_sde_state_lpdf_for`: Same but with `state_lpdf()` checking for-loop against vmap.

## Improvements

- `test_{method}_models`: Test that given method (e.g., `state_lpdf()`, `pf_step()`, etc.) of two models give the same results.

- `test_{method}_jit`: Test that given method (e.g., `state_lpdf()`, `pf_step()`, etc.) is same with and without JIT, and also for grads.

# MVN Bridge Tests

- `test_tri_fact()`: Test that the three-term factorization `p(W) p(X | W) p(Y | X)` underlying the general MVN bridge proposal is equal to the joint MVN `p(W, X, Y)` on all terms.
- `test_double_fact()`: Test that the two term factorization `p(Y) p(X | Y)` is equal to the joint MVN `p(X, Y)`.

# Contents of `pfjax/src/pfjax/test`

This folder contains the test functions living inside the package, i.e., those which are available via `import pfjax.test`.  These are mostly the for-loop versions of the generic methods (`simulate()`, `loglik_full()`, etc.), and are ostensibly there in order to simplify interactive package development.  These can probably be safely moved to outside of the package to `pfjax/tests` if we are careful with the import layout.

