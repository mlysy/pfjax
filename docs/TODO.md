# Documentation

- [x] Fix relative links in `README.md` included by `index.md`.

	Did this by putting files to link to in `docs`.

	More info [here](https://myst-parser.readthedocs.io/en/v0.13.5/using/howto.html#include-a-file-from-outside-the-docs-folder-like-readme-md).
	
- [ ] Display of return values (colons are problematic).

	**Update:** Have done some of this but not all.
	
- [ ] `autoapi` index page looks wrong.

- [x] Where to put "non-exported" notebooks?  Natural place is e.g., `docs/devel`, but `myst-nb` renders all notebooks in `docs` and its subfolders.

	They are now in `docs/notebooks/internal`.  Skip rendering by adding to `exclude_patterns` in `conf.py`.

- [x] Fix math rendering in jupyter notebooks.  Ideally we would like to use the latex macros defined in `latexdefs.tex` using the `latex_envs` jupyter extension as documented [here](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/latex_envs/README.html).  

	The solution was to parse `latexdefs.tex` into Mathjax.  So far seems to work for the macros defined, but will likely need to extend later.
	
- [x] Fix references in jupyter notebooks.  Ideally would citations defined in `biblio.tex` using `latex_envs`.

	~~This is because `latex_envs` adds a reference section directly to the notebook (which renders fine) but leaves the `\cite{}` commands in the Markdown sections as-is (rendering them with something else).  So we need to figure out how to apply this renderer.  Perhaps possible via method described [here](https://myst-nb.readthedocs.io/en/latest/authoring/custom-formats.html#custom-formats).~~
	
	**Update:** Fixed this by switching long-form documentation format to [MyST](https://myst-parser.readthedocs.io/en/latest/index.html).  This is basically `.md` files with many useful extensions.  You can also pair these with Jupyter Notebooks using [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html), so that developing the documentation can be done with Jupyter Notebook if desired.  The downside is that this isn't compatible with `latex_envs` for references.

- [ ] Put the bibliography at the end of each file.  Currently just one bibliography at the end of everything.  Perhapse using [`footcite`](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#section-local-bibliographies)?

- [ ] Get `version/release` and `author` info from `setup.cfg`.

	Have attempted to do this using [this](https://github.com/pypa/setuptools/issues/2530#issuecomment-1135391647) and [this](https://stackoverflow.com/questions/26141851/let-sphinx-use-version-from-setup-py).
	
	However, version in which info is stored in a separate file `src/pfjax/__metadata__.py` does not work.  It only works if `__version__` and `__author__` are defined directly in `src/pfjax/__init__.py`.
	
	Tested that the above does work for `docs`.

- [ ] Clean up a ton of warnings when running `make html`.  


# Code

- [ ] Use Quarto for development-side documentation (so much easier to write).  Here's what could be involved:

	1. Convert all (or most of) the `.md` and `.ipynb` files to `.qmd`.  This can be done with **jupytext**.
	
	2. Reformat all docstrings to Quarto Markdown.  For example:
	
		- Inline code is written with single instead of double backticks.
		
		- Latex math is declared differently.  However, I've come to the conclusion that it's not a good idea to use latex in function documentation, because the source code is often illegible.  Instead, try to avoid math in docstrings whenever possible.  If you absolutely need it, then best to use pseudocode math in a fenced environment (i.e., renders as code).
		
		- Please use **numpydoc** style for documentation.  It's a bit more verbose than what we're using now.
		
		- Probably best to do all this for one file first to iron out the "template" for all the others.
		
	3. Use Quarto and **quartodocs** to render the documentation.
	
	4. The output of step 3 is functional but, in my opinion, somewhat limited in usability.  A more useful output format is that of **mkdocs** or **sphinx**.  However, there are no existing tools out there to convert from Quarto input to either of those output formats.

- [ ] Refactor unit tests to use **pytest**.

- [ ] Systematically deal with prior on initial state.  For example, it is missing from `loglik_full()`.  It is also missing from gradient/hessian calculations in `particle_filter()` and `particle_filter_rb()`.

- [ ] Naming of things:

	- [ ] Change `*_sample()` to `*_sim()`.
	
	- [ ] Change `{x/y}_init` to `{x/y}_curr`.  Or maybe not?  Which is more likely to improve the user's experience?  Probably the latter, right?

	- [x] `{prior/state/meas}_{lpdf/sim}()`: Define the relevant pieces of the state-space model itself.
	
		Done the first part but haven't changed `sample` to `sim`.
	
	- [ ] `{init/step}_{lpdf/sim}()`: Define the relevant pieces of the proposal distribution.

	- [ ] `{init/step}_particle()`: Combinations of the above to return a particle and its (unnormalized) log-weight.  This can be done automatically, but users can define these manually if there are lots of duplicate calculations in doing it automatically. 
	
		**Note:** There are currently called `pf_init()` and `pf_step()`.
		
		**Edit:** Can't we just call these `init()` and `step()`?

	- [ ] `particle_filter = AuxillaryPF()`: Then just run it as a functor.  Could do the same with `particle_filter = RaoBlackwellPF()`.
	
		```python
		# current api
		output = particle_filter(model=model, key=key, ...) # usual particle filter
		output2 = particle_filter_rb(model=model, key=key, ...) # rao-blackwellized pf
		output3 = particle_filter_xyz(...) # some other pf we'll eventually define
		
		# non-functor api
		class AuxillaryPF():
	        def __call__(self, key, model, ...):
			    return particle_filter(model=model, key=key, ...)
		
		if selected_filter == "aux":
	        filter = AuxillaryPF(gradient=True, history=False)
		elif selected_filter == "rb":
		    filter = RaoBlackwellPF()

	    output = filter(model=model, key=key, ...)
		```


- [ ] Use **numpyro** to create `*_{lpdf/sim}()`.  See `tests/interactive/test_numpyro.py`.

- [ ] Maybe more easily, could have a specific interface to create `*_{lpdf/sim}()` for a normal distribution, which is the most common application of this.
