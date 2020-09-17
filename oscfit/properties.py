"""Definition of project-global variables"""

import numpy as np
# Global colors
from colors import dark2


# Data properties
data_dir = '../data'
plot_dir = '../figures'
model_dir = '../models'
results_dir = '../results'
# Plot properties
figsize = (7, 7)
fontsize = 14
# Data generation properties
parameter_ranges = {
    'gamma': (100, 700),
    'mass': (0.01, 100),
    'freq': (500, 6400)
}
n_samples_per_class = 10000
n_osc_classes = (2, 3)
x_range = np.arange(500, 6501, 10)

