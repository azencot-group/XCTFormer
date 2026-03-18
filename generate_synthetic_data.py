"""Generate the Multivariate Dependent Synthetic dataset.

Run this script before training on the Synthetic dataset:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --seed 2021 --n_points 10000

The generated CSV will be saved to datasets/Synthetic/synthetic.csv
"""

import os
import argparse
from data_provider.synthetic_data import generate_synthetic_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--n_points', type=int, default=10000)
    args = parser.parse_args()

    output_dir = os.path.join('datasets', 'Synthetic')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'synthetic.csv')
    generate_synthetic_csv(output_path, n_points=args.n_points, seed=args.seed)
