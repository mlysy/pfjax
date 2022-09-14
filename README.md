# **pfjax**: Particle Filtering in JAX

---

## What is pfjax?

- An [introduction](docs/notebooks/pfjax.ipynb) to **pjfax**.

## Installation

This will clone the repo into a subfolder `pfjax` of where you issue the `git clone` command, then install the package from source.

```bash
git clone https://github.com/mlysy/pfjax
cd pfjax
pip install .
``` 

## Quickstart 
- An [example](docs/notebooks/lotvol.ipynb) of inference with **pfjax** for a stochastic differential equation model.

### Brownian Motion


## Testing

From within `pfjax/tests`:

```bash
python3 -m unittest -v
```

Or, install [**tox**](https://tox.wiki/en/latest/index.html), then from within `pfjax` at the command line: `tox`.

## Documentation

From within `pfjax/docs`:

```bash
make html
# or alternatively
make clean html
```

## Reference docs

More examples and further documentation can be found here:
