# **pfjax**: Particle Filtering in JAX

---

## Introduction
- An [introduction](examples/pfjax.ipynb) to **pjfax**.

## Installation

This will clone the repo into a subfolder `pfjax` of where you issue the `git clone` command, then install the package from source.

```bash
git clone https://github.com/mlysy/pfjax
cd pfjax
pip install .
```

## Examples
- An [example](examples/lotvol.ipynb) of inference with **pfjax** for a stochastic differential equation model.

### Brownian Motion



More examples and further documentation can be found here: 

## Testing

From within `pfjax/tests`:

```bash
python3 -m unittest -v
```

Or, install [**tox**](https://tox.wiki/en/latest/index.html), then from within `pfjax` enter command line: `tox`.