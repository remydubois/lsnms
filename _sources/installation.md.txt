# Installation

LSNMS is fully installable with pip: this shall downloaded pre-compiled binaries for the rust backend.
```
pip install lsnms --upgrade
```
or by cloning this repo with poetry and installing both the python package and compile the rust crate:
```
git clone https://github.com/remydubois/lsnms
cd lsnms/
make install
```
Only dependencies are numpy and numba.