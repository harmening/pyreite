# :skull_and_crossbones: pyreite
# Pythonic, Yet Rudimental, Electrical Impedance Tomography Expert
**Code snippets for pythonic head modeling with Electrical Impedance Tomography (EIT).**<br>
<br>
[![build](https://github.com/harmening/pyreite/actions/workflows/action.yml/badge.svg)](https://github.com/harmening/pyreite/actions)
[![coverage](https://codecov.io/gh/harmening/pyreite/branch/main/graph/badge.svg?token=LHJ5W57UE8)](https://codecov.io/gh/harmening/pyreite)
[![python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)



## Get up and running
### Prerequisites
- [Python3.6](https://www.python.org/downloads/)
- [OpenMEEG](https://github.com/openmeeg/openmeeg/blob/master/README.rst#build-openmeeg-from-source) 2.4 with python wrapping: compile with `"-DENABLE_PYTHON=ON"`

### Install pyreite
```bash
git clone https://gitlab.tubit.tu-berlin.de/promillenille/pyreite
cd pyreite
pip install -r requirements.txt
python setup.py install
```


## docker :whale:
Build pyreite image
```bash
$ docker build -t pyreite .
```
or pull from [docker hub](https://hub.docker.com/r/harmening/pyreite)
```bash
$ docker pull harmening/pyreite:v0.2
```
<br>



