# :skull_and_crossbones: pyreite
# Pythonic, Yet Rudimental, Electrical Impedance Tomography Expert
**Code snippets for pythonic head modeling with Electrical Impedance Tomography (EIT).**<br>
<br>
[![build](https://github.com/harmening/pyreite/actions/workflows/action.yml/badge.svg)](https://github.com/harmening/pyreite/actions)
[![codecov](https://codecov.io/gh/harmening/pyreite/graph/badge.svg?token=ASuSRfHmV1)](https://codecov.io/gh/harmening/pyreite)
[![python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)



## Get up and running
### Prerequisites
- [Python3.6](https://www.python.org/downloads/)
- [OpenMEEG](https://github.com/openmeeg/openmeeg/blob/master/README.rst#build-openmeeg-from-source) 2.4 with python wrapping: compile with `"-DENABLE_PYTHON=ON"`

### Install pyreite
```bash
git clone https://github.com/harmening/pyreite.git
cd pyreite
pip install -r requirements.txt
python setup.py install
```


### docker :whale:
Build pyreite image
```bash
$ docker build -t pyreite .
```
or pull from [docker hub](https://hub.docker.com/r/harmening/pyreite)
```bash
$ docker pull harmening/pyreite:v0.2
```
<br>


## Example EIT simulation
```bash
import os.path.join as pth
from collections import OrderedDict
from pyreite.OpenMEEGHead import OpenMEEGHead
from pyreite.data_io import load_tri, load_elecs_dips_txt


# Load Colins surface meshes
geom = OrderedDict()
for tissue in ['cortex', 'csf', 'skull', 'scalp']:
    geom[tissue] = load_tri(pth('tests', 'test_data', tissue+'.tri'))

# Define conductivity values [S/m]
cond = {'cortex': 0.201, 'csf': 1.65, 'skull': 0.01, 'scalp': 0.465}

# Load electrode positions
sens = load_elecs_dips_txt(pth('tests', 'test_data', 'electrodes_aligned.txt'))


# Create EIT head model
model = OpenMEEGHead(cond, geom, sens)

# Calculate EIT voltage measurement array
V = model.V
```
<br>
