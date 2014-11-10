# Functions to load EEG data from .mat files.

import os.path
import numpy as np
from scipy.io import loadmat


def load_data(filename):
    """
    Read data from a .mat file and return the relevant parts
    in a dictionary.
    """
    mat = loadmat(os.path.abspath(filename))
    for key in mat.keys():
        if 'segment' in key:
            segment = mat[key][0][0]
            if key.startswith('pre'):
                segtype = 'preictal'
            elif key.startswith('inter'):
                segtype = 'interictal'
            elif key.startswith('test'):
                segtype = 'test'
            else:
                print 'Warning: unrecognized segment type ' + key
                segtype = None

    hour = int(filename.split('_')[-1].split('.')[0])/6

    data = {'data': segment[0],
            'length_sec': segment[1][0][0],
            'sampling_rate_hz': segment[2][0][0],
            'names': [str(segment[3][0][i][0]) for i in \
                      range(len(segment[3][0]))],
            'type': segtype,
            'hour': hour}
    if segtype != 'test':
        data['hour_index'] = segment[4][0][0]
    else:
        data['hour_index'] = np.nan

    return data

    
