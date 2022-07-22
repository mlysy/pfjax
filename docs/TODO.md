
- [x] Fix relative links in `README.md` included by `index.md`.

	Did this by putting files to link to in `docs`.

	More info [here](https://myst-parser.readthedocs.io/en/v0.13.5/using/howto.html#include-a-file-from-outside-the-docs-folder-like-readme-md).
	
- [ ] Display of return values (colons are problematic).

- [x] Where to put "non-exported" notebooks?  Natural place is e.g., `docs/devel`, but `myst-nb` renders all notebooks in `docs` and its subfolders.

	THey are now in `docs/notebooks/internal`.  Skip rendering by adding to `exclude_patterns` in `conf.py`.

- [ ] Fix math rendering in jupyter notebooks.  Ideally we would like to use the latex macros and citations defined in `latexdefs.tex` and `biblio.bib` respectively, using `jupyter-contrib-nbextensions` as documented [here](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/latex_envs/README.html).  Note however that while the macros at least render fine when you open a regular jupyter notebook, I never got the citations to work.  And neither of these currently work in the rendered readthedocs. 
