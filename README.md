# Pythonic, Yet Rudimental, Electrical Impedance Tomography Expert (Pyreite)  :skull_and_crossbones:  :skull_and_crossbones:  :skull_and_crossbones:

[![build](https://github.com/harmening/pyreite/actions/workflows/action.yml/badge.svg)](https://github.com/harmening/pyreite/actions)
[![codecov](https://codecov.io/gh/harmening/pyreite/graph/badge.svg?token=ASuSRfHmV1)](https://codecov.io/gh/harmening/pyreite)
[![python](https://img.shields.io/badge/python-3.7|3.8|3.9|3.10|3.11-blue.svg)](https://www.python.org/downloads/release/python-360/)
<img align="right" width="330" src="logo.png"> <br>


### Pythonic head model algorithms for Electrical Impedance Tomography (EIT)<br>
With only small changes to the hardware (no changes to the EEG cap setup), an EIT measurement can be performed in less than 60 seconds before running the actual EEG experiment.<br>
Pyreite aims to derive individual electrical conductivity parameters for the different biological head tissues (scalp, skull, CSF, cortex) from that EIT data.
Knowing these subject-dependent (and also within subject-dependent) parameters **can significantly improve EEG source localization accuracy**!
<br>
<br>




## Get up and running
### Prerequisites
- [Python3](https://www.python.org/downloads/)
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
