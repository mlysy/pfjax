
- [x] Fix relative links in `README.md` included by `index.md`.

	Did this by putting files to link to in `docs`.

	More info [here](https://myst-parser.readthedocs.io/en/v0.13.5/using/howto.html#include-a-file-from-outside-the-docs-folder-like-readme-md).
	
- [ ] Display of return values (colons are problematic).

- [x] Where to put "non-exported" notebooks?  Natural place is e.g., `docs/devel`, but `myst-nb` renders all notebooks in `docs` and its subfolders.

	They are now in `docs/notebooks/internal`.  Skip rendering by adding to `exclude_patterns` in `conf.py`.

- [x] Fix math rendering in jupyter notebooks.  Ideally we would like to use the latex macros defined in `latexdefs.tex` using the `latex_envs` jupyter extension as documented [here](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/latex_envs/README.html).  

	The solution was to parse `latexdefs.tex` into Mathjax.  So far seems to work for the macros defined, but will likely need to extend later.
	
- [x] Fix references in jupyter notebooks.  Ideally would citations defined in `biblio.tex` using `latex_envs`.

	~~This is because `latex_envs` adds a reference section directly to the notebook (which renders fine) but leaves the `\cite{}` commands in the Markdown sections as-is (rendering them with something else).  So we need to figure out how to apply this renderer.  Perhaps possible via method described [here](https://myst-nb.readthedocs.io/en/latest/authoring/custom-formats.html#custom-formats).~~
	
	**Update:** Fixed this by switching long-form documentation format to [MyST](https://myst-parser.readthedocs.io/en/latest/index.html).  This is basically `.md` files with many useful extensions.  You can also pair these with Jupyter Notebooks using [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html), so that developing the documentation can be done with Jupyter Notebook if desired.  The downside is that this isn't compatible with `latex_envs` for references.


- [ ] Get `version/release` and `author` info from `setup.cfg`.

	Have attempted to do this using [this](https://github.com/pypa/setuptools/issues/2530#issuecomment-1135391647) and [this](https://stackoverflow.com/questions/26141851/let-sphinx-use-version-from-setup-py).
	
	However, version in which info is stored in a separate file `src/pfjax/__metadata__.py` does not work.  It only works if `__version__` and `__author__` are defined directly in `src/pfjax/__init__.py`.
	
	Also, haven't yet tested whether this works for `docs`.

- [ ] Clean up a ton of warnings when running `make html`.  
