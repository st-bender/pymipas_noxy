# PyMIPAS_NOxy

**MIPAS NOx/NOy calculator tools**

Some tools to handle MIPAS level-2 netCDF files,
and to combine trace gase species to calculate
NOx (= NO + NO2) and NOy (= NO + NO2 + ClONO2 + N2O5 + HNO3 + HNO4)
from the single species.

:warning: This package is in **alpha** stage, that is, it works mostly,
but the interface might still be subject to change.

## Install

### Requirements

- `numpy` - required
- `xarray` - required for reading netCDF files
- `netCDF4` - required for reading netCDF files
- `scipy` - required for the interpolation interface
- `click` - required for the command line interface
- `toml` - required to read the configuration files
- `pytest` - optional, for testing

### mipas_noxy

An installable `pip` package called `mipas_noxy` will soon be available
from the main package repository, it can then be installed with:
```sh
$ pip install mipas_noxy
```
The latest development version can be installed
with [`pip`](https://pip.pypa.io) directly from github
(see <https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support>
and <https://pip.pypa.io/en/stable/reference/pip_install/#git>):

```sh
$ pip install [-e] git+https://github.com/st-bender/pymipas_noxy.git
```

The other option is to use a local clone:

```sh
$ git clone https://github.com/st-bender/pymipas_noxy.git
$ cd pymipas_noxy
```
and then using `pip` (optionally using `-e`, see
<https://pip.pypa.io/en/stable/reference/pip_install/#install-editable>):

```sh
$ pip install [-e] .
```

or using `setup.py`:

```sh
$ python setup.py install
```

Optionally, test the correct function of the module with

```sh
$ py.test [-v]
```

or even including the [doctests](https://docs.python.org/library/doctest.html)
in this document:

```sh
$ py.test [-v] --doctest-glob='*.md'
```

## Usage

The python module itself is named `mipas_noxy` and is imported as usual.

All functions should be `numpy`-compatible and work with scalars
and appropriately shaped arrays.

```python
>>> import mipas_noxy as moxy

```

Basic class and method documentation is accessible via `pydoc`:

```sh
$ pydoc mipas_noxy
```

## License

This python interface is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 (GPLv2), see [local copy](./LICENSE)
or [online version](http://www.gnu.org/licenses/gpl-2.0.html).
