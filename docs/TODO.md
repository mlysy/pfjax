
- [x] Fix relative links in `README.md` included by `index.md`.

	Did this by putting files to link to in `docs`.

	More info [here](https://myst-parser.readthedocs.io/en/v0.13.5/using/howto.html#include-a-file-from-outside-the-docs-folder-like-readme-md).
	
- [ ] Display of return values (colons are problematic).

- [x] Where to put "non-exported" notebooks?  Natural place is e.g., `docs/devel`, but `myst-nb` renders all notebooks in `docs` and its subfolders.

	THey are now in `docs/notebooks/internal`.  Skip rendering by adding to `exclude_patterns` in `conf.py`.

- [ ] Fix math rendering in jupyter notebooks.  Ideally we would like to use the latex macros and citations defined in `latexdefs.tex` and `biblio.bib` respectively, using `jupyter-contrib-nbextensions` as documented [here](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/latex_envs/README.html).  Note however that while the macros at least render fine when you open a regular jupyter notebook, I never got the citations to work.  And neither of these currently work in the rendered readthedocs. 

	The math part is semi-fixed.  Only `\begin{aligned}` still not working.

- [ ] Get `version/release` and `author` info from `setup.cfg`.

	Have attempted to do this using [this](https://github.com/pypa/setuptools/issues/2530#issuecomment-1135391647) and [this](https://stackoverflow.com/questions/26141851/let-sphinx-use-version-from-setup-py).
	
	However, version in which info is stored in a separate file `src/pfjax/__metadata__.py` does not work.  It only works if `__version__` and `__author__` are defined directly in `src/pfjax/__init__.py`.
	
	Also, haven't yet tested whether this works for `docs`.
