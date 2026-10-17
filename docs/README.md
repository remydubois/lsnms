Building the documentation
==========================

Prerequisites
-------------

Install the required packages (preferably in a virtualenv):

```bash
poetry install --with dev
```

Build
-----

From the repository root run:

```bash
make docs-python
```

The generated site will be in `docs/_build/html`.
