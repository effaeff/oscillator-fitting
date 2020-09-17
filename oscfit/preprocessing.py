"""Random generation of data for oscillator parameter values"""

import random
import numpy as np

from oscillator import calc_frf

from properties import (
    data_dir,
    parameter_ranges,
    n_samples_per_class,
    n_osc_classes,
    x_range
)


class PreProcessor:
    """Class to wrap data generation, preprocessing and data storing"""
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config['batch_size']


    def generate_data(self):
        """Generation method"""
        for n_osc in n_osc_classes:
            data_inp = np.empty((n_samples_per_class, 3, len(x_range)))
            data_out = np.empty((n_samples_per_class, n_osc * 3))
            for idx_sample in range(n_samples_per_class):
                params = np.empty(n_osc * 3)
                for idx_osc in range(n_osc):
                    params[idx_osc * 3:(idx_osc + 1) * 3] = [
                        random.uniform(parameter_ranges['freq'][0], parameter_ranges['freq'][1]),
                        random.uniform(parameter_ranges['gamma'][0], parameter_ranges['gamma'][1]),
                        random.uniform(parameter_ranges['mass'][0], parameter_ranges['mass'][1])
                    ]

                amp, phase = calc_frf(x_range, params)

                data_inp[idx_sample] = np.array([x_range, amp, phase])
                data_out[idx_sample] = params
            np.save(f'{data_dir}/{n_osc}osc_inp.npy', data_inp)
            np.save(f'{data_dir}/{n_osc}osc_out.npy', data_out)

