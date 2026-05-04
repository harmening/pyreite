import numpy as np
from pyreite.EIThelpers import EIT_protocol


def test_EIT_protocol():
    num_elec = np.random.randint(100)
    ND2V = EIT_protocol(num_elec, n_freq=1, protocol='all_realistic')
    assert np.sum(ND2V) == num_elec*(num_elec-1)/2 * (num_elec-2)
    assert len(ND2V) == num_elec**3
    n_freq = np.random.randint(10)
    ND2V = EIT_protocol(num_elec, n_freq=n_freq, protocol='all')
    assert len(ND2V) == np.sum(ND2V) == (n_freq*num_elec**3)
