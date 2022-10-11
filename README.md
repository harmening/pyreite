# :skull_and_crossbones: pyreite
# Pythonic, Yet Rudimental, Electrical Impedance Tomography Expert
**Code snippets for pythonic head modeling with Electrical Impedance Tomography (EIT).**<br>



## Get up and running
### Prerequisites
- [Python3.6](https://www.python.org/downloads/) or newer
- [OpenMEEG](https://github.com/openmeeg/openmeeg/blob/master/README.rst#build-openmeeg-from-source) with python wrapping: compile with `"-DENABLE_PYTHON=ON"`

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
$ docker pull harmening/pyreite
```
<br>



