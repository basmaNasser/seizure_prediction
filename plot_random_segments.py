#!/usr/bin/env python
#
# Randomly draw one preictal segment and one interictal segment
# and, for each, plot the time series and power spectrum from a 
# single, randomly-chosen electrode.

import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import load_data

data_dir = os.path.abspath('data/Dog_1')

# width of Gaussian filter in seconds
filter_width = 0.0

# load random preictal and interictal segments
pre_data, inter_data = [load_data.load_random_data(data_dir, seg_type,
                                                   print_file_name=True) for \
                            seg_type in ('preictal', 'interictal')]

# choose random electrodes for each segment
pre_shape = pre_data['data'].shape
pre_elec = np.random.choice(pre_shape[0])
inter_shape = inter_data['data'].shape
inter_elec = np.random.choice(inter_shape[0])

# set up plot
fig = plt.figure(figsize=(8,8))
fig.set_tight_layout(True)

plt.subplot(221)
voltage = pre_data['data'][pre_elec,:]
if filter_width > 0:
    voltage = gaussian_filter1d(voltage,
                                filter_width*pre_data['sampling_rate_hz'],
                                mode='constant')
plt.plot(np.linspace(0, pre_data['length_sec'], num=pre_shape[1]),
         voltage, 'k-')
plt.xlabel('time [s]')
plt.ylabel('preictal voltage')

plt.subplot(222)
freq = np.fft.rfftfreq(voltage.size, d=1./pre_data['sampling_rate_hz'])
power = np.abs(np.fft.rfft(voltage))**2
plt.loglog(freq, power, 'k-')
plt.xlabel('frequency [Hz]')
plt.ylabel('preictal power')

plt.subplot(223)
voltage = inter_data['data'][inter_elec,:]
if filter_width > 0:
    voltage = gaussian_filter1d(voltage,
                                filter_width*inter_data['sampling_rate_hz'],
                                mode='constant')
plt.plot(np.linspace(0, inter_data['length_sec'], num=inter_shape[1]),
         voltage, 'k-')
plt.xlabel('time [s]')
plt.ylabel('interictal voltage')

plt.subplot(224)
freq = np.fft.rfftfreq(voltage.size, d=1./inter_data['sampling_rate_hz'])
power = np.abs(np.fft.rfft(voltage))**2
plt.loglog(freq, power, 'k-')
plt.xlabel('frequency [Hz]')
plt.ylabel('interictal power')

plt.show()
