pfjax: Particle Filtering in JAX
=================================

==============================
What is pfjax?
==============================

The goal is to provide a fast implementation of a particle filter to estimate the marginal likelihood of a state-space model along with its gradient.

The JAX library is used to efficiently compose jit and autodiff operations in order to achieve this. 

**Insert overview of the purpose of the package and definitions here**

==============================
Installation
==============================

This will clone the repo into a subfolder ``pfjax``  of where you issue the ``git clone`` command, then install the package from source.

.. code-block:: sh
   git clone https://github.com/mlysy/pfjax
   cd pfjax
   pip install .

==============================
Quickstart
==============================

Brownian Motion Example
========================

Maximum Likelihood Estimators
******************************

Variance Estimators
*********************

==============================
Testing
==============================
From within ``pfjax/tests``:

.. code-block:: sh
   python3 -m unittest -v

Or you can install `tox <https://tox.wiki/en/latest/index.html>`_, then from within ``pfjax`` enter the command line: ``tox``.

==============================
Function Documentation
==============================

.. toctree::
   :maxdepth: 1

   source/pfjax

==============================
More Use Cases
==============================
.. nbgallery::

   ..
      Input the notebook file names here:
      notebooks/mcmc
      notebooks/pgnet
      notebooks/sde_module

==============================
Links
==============================
* :ref:`search`
* Source code: https://github.com/mlysy/pfjax
* pages/faq