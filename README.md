# **PFJAX**: Particle Filtering in JAX

[**Quickstart**](docs/notebooks/pfjax.md)
| [**Installation**](#installation)
| [**Documentation**](#documentation)
| [**Developers**](#developers)

---

## What is **PFJAX**?

**PFJAX** is a collection of tools for estimating the parameters of state-space models using particle filtering methods, with  [**JAX**](https://jax.readthedocs.io/) as the backend for JIT-compiling models and automatic differentiation.

---

## Installation

This will clone the repo into a subfolder `pfjax`, from where you (i) issue the `git clone` command and (ii) install the package from source.

```bash
git clone https://github.com/mlysy/pfjax
cd pfjax
pip install .
``` 

## Quickstart 

A brief [introduction to **PFJAX**](docs/notebooks/pfjax.md).

## Documentation

This is a work in progress!  Current modules include:

- The [quickstart guide](docs/notebooks/pfjax.md).

- A [comparison of gradient and hessian algorithms](docs/notebooks/gradient_comparisons.md) based on particle filters, which in turn are used for conducting parameter inference.

- An example of parameter inference using [stochastic optimization](docs/notebooks/stochopt_tutorial.md).

- An example of parameter inference using [Markov chain Monte Carlo](docs/notebooks/mcmc_tutorial).

- The API [reference documentation](https://pfjax.readthedocs.io/).

## Developers

### Testing

From within `pfjax/tests`:

```bash
python3 -m unittest -v
```

Or, install [**tox**](https://tox.wiki/en/latest/index.html), then from within `pfjax` at the command line: `tox`.

### Building Documentation

From within `pfjax/docs`:

```bash
# regular build
make html

# clean build incl. repeating cached computations
make clean html
```
