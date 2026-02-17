# PFJAX: Particle Filtering in JAX

[**Quickstart**](#quickstart)
| [**Installation**](#installation)
| [**Documentation**](#documentation)
| [**Developers**](#developers)

---

## What is PFJAX?

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

A brief [introduction to **PFJAX**](https://mlysy.github.io/pfjax/pfjax/).

## Documentation

This is a work in progress!  Current modules include:

- The [quickstart guide](https://mlysy.github.io/pfjax/pfjax/).

- A [comparison of gradient and hessian algorithms](https://mlysy.github.io/pfjax/gradient_comparisons/) based on particle filters, which in turn are used for conducting parameter inference.

- An example of parameter inference using [stochastic optimization](https://mlysy.github.io/pfjax/stochopt_tutorial/).

- An example of parameter inference using [Markov chain Monte Carlo](https://mlysy.github.io/pfjax/mcmc_tutorial/).

- The API [reference documentation](https://mlysy.github.io/pfjax/reference/pfjax/).

## Developers

The instructions below assume that [**uv**](https://docs.astral.sh/uv/) is being used for dependency management.  Instructions for installing **uv** are available [here](https://docs.astral.sh/uv/getting-started/installation/).

### Testing

From within the `pfjax` folder:

```bash
uv run --group test pytest
```

### Building Documentation

The documentation is build using [**Quarto**](https://quarto.org/) + [**MkDocs-Material**](https://squidfunk.github.io/mkdocs-material/).  The latter comes as a Python package installed by **uv**, but the former must be [installed](https://quarto.org/docs/get-started/) separately.  Once **Quarto** is installed, from within the `pfjax` folder:

```bash
uv run --group docs quarto render docs/
uv run --group docs mkdocs build
```
