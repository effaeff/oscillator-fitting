"""Main script"""

import os
import numpy as np

import misc
from generate_data import generate_data
from properties import (
    data_dir,
    plot_dir,
    results_dir,
    model_dir
)

def main():
    """Main method"""
    misc.to_local_dir(__file__)
    misc.gen_dirs([data_dir, plot_dir, model_dir, results_dir])
    generate_data()

if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()

