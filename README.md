# **pfjax**: Particle Filtering in JAX

---

## Installation

This will clone the repo into a subfolder `pfjax` of where you issue the `git clone` command, then install the package from source.

```bash
git clone https://github.com/mlysy/pfjax
cd pfjax
pip install .
```

## Documentation

- An [introduction](examples/pfjax.ipynb) to **pjfax**.

- An [example](examples/lotvol.ipynb) of inference with **pfjax** for a stochastic differential equation model.

## Testing

From within `pfjax/tests`:

```bash
python3 -m unittest -v
```

