"""
Multivariate Dependent Synthetic Data Generator.

Generates patch-dependent multivariate time series where the target variate
depends on cross-variate patch-level relationships (Granger causality).

Design:
- Univariate models FAIL: target cannot be predicted from its own past alone
- Channel-independent models FAIL: patches from different variates must interact
- Cross-variate attention models SUCCEED: they learn pairwise patch relationships

The target patches are blends of source patches, where blend weights follow
an oscillating triangle wave pattern (period 20 patches).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SignalConfig:
    """Configuration for a base signal (source variate)."""
    signal_type: str  # 'sine' or 'random_walk'
    # Sine parameters
    amplitude: float = 1.0
    frequency: float = 0.02
    phase: float = 0.0
    # Random walk parameters
    step_std: float = 0.1
    # Common
    noise_std: float = 0.02


@dataclass
class PatchPairConfig:
    """Configuration for a pairwise patch relationship.

    The target patch is influenced by the relationship between patches
    from source_var_i and source_var_j at specified lags.
    """
    source_var_i: int
    source_var_j: int
    lag_i: int = 1
    lag_j: int = 1
    weight: float = 1.0


@dataclass
class SyntheticConfig:
    """Full configuration for the synthetic dataset."""
    n_source_variates: int
    patch_len: int = 16
    stride: int = 8
    patch_pairs: List[PatchPairConfig] = None
    base_signals: List[SignalConfig] = None
    noise_std: float = 0.02
    seed: Optional[int] = None


def get_synthetic_config() -> SyntheticConfig:
    """Get the default synthetic dataset configuration.

    6 source variates (4 sine waves + 2 random walk distractors) -> target.
    Blend weights follow an oscillating triangle wave with period 20 patches
    (10 up + 10 down), stepping by 0.1.
    """
    return SyntheticConfig(
        n_source_variates=6,
        patch_len=16,
        stride=8,
        patch_pairs=[
            PatchPairConfig(
                source_var_i=0,  # sine1
                source_var_j=1,  # sine2
                lag_i=1,
                lag_j=2,
                weight=0.5,
            ),
            PatchPairConfig(
                source_var_i=2,  # sine3
                source_var_j=3,  # sine4
                lag_i=2,
                lag_j=3,
                weight=0.5,
            ),
        ],
        base_signals=[
            SignalConfig(signal_type='sine', amplitude=1.0, frequency=0.02, noise_std=0.02),
            SignalConfig(signal_type='sine', amplitude=3.0, frequency=0.03, phase=1.0, noise_std=0.07),
            SignalConfig(signal_type='sine', amplitude=2.0, frequency=0.01, phase=1.0, noise_std=0.03),
            SignalConfig(signal_type='sine', amplitude=5.0, frequency=0.002, phase=0.5, noise_std=0.02),
            SignalConfig(signal_type='random_walk', step_std=0.1, noise_std=0.0),
            SignalConfig(signal_type='random_walk', step_std=0.15, noise_std=0.0),
        ],
        noise_std=0.02,
        seed=2021,
    )


class SyntheticDataGenerator:
    """Generator for patch-dependent time series with Granger causality structure.

    Target patches are blends of source patches from other variates.
    Blend weights oscillate as a triangle wave over patch indices.
    """

    def __init__(self, config: SyntheticConfig):
        self.config = config

    def generate(self, n_points: int) -> np.ndarray:
        """Generate multivariate time series.

        Returns:
            data: np.ndarray of shape [n_points, n_source_variates + 1]
                  Last column is the target variate.
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        n_variates = self.config.n_source_variates + 1  # +1 for target
        data = np.zeros((n_points, n_variates))

        # Generate source variates
        for i in range(self.config.n_source_variates):
            cfg = self.config.base_signals[i]
            data[:, i] = self._generate_signal(n_points, cfg)

        # Generate target based on patch relationships
        data[:, -1] = self._generate_target(data[:, :-1], n_points)

        return data

    def _generate_signal(self, n_points: int, cfg: SignalConfig) -> np.ndarray:
        t = np.arange(n_points)
        if cfg.signal_type == 'sine':
            data = cfg.amplitude * np.sin(2 * np.pi * cfg.frequency * t + cfg.phase)
        elif cfg.signal_type == 'random_walk':
            steps = np.random.normal(0, cfg.step_std, n_points)
            data = np.cumsum(steps)
        else:
            raise ValueError(f"Unknown signal type: {cfg.signal_type}")

        if cfg.noise_std > 0:
            data += np.random.normal(0, cfg.noise_std, n_points)
        return data

    def _generate_target(self, source_data: np.ndarray, n_points: int) -> np.ndarray:
        """Generate target from pairwise patch relationships."""
        patch_len = self.config.patch_len
        stride = self.config.stride
        patch_num = (n_points - patch_len) // stride + 1

        target = np.zeros(n_points)
        counts = np.zeros(n_points)

        for k in range(patch_num):
            target_patch = np.zeros(patch_len)
            total_weight = 0.0

            for pair in self.config.patch_pairs:
                patch_i = self._extract_patch(source_data, pair.source_var_i, k - pair.lag_i)
                patch_j = self._extract_patch(source_data, pair.source_var_j, k - pair.lag_j)

                if patch_i is None or patch_j is None:
                    continue

                # Oscillating blend weight: triangle wave 0->1->0->1...
                # Period = 20 patches (10 up + 10 down)
                blend_weight = self._oscillating_weight(k)

                # Target patch is weighted combination of source patches
                blended_patch = blend_weight * patch_i + (1 - blend_weight) * patch_j
                target_patch += pair.weight * blended_patch
                total_weight += pair.weight

            if total_weight > 0:
                target_patch /= total_weight

            start = k * stride
            end = start + patch_len
            target[start:end] += target_patch
            counts[start:end] += 1

        # Average overlapping regions
        target = target / np.maximum(counts, 1)
        # Add noise
        target += np.random.normal(0, self.config.noise_std, n_points)
        return target

    def _extract_patch(self, data: np.ndarray, var_idx: int, patch_idx: int):
        if patch_idx < 0:
            return None
        start = patch_idx * self.config.stride
        end = start + self.config.patch_len
        if end > data.shape[0]:
            return None
        return data[start:end, var_idx].copy()

    @staticmethod
    def _oscillating_weight(patch_idx: int) -> float:
        """Triangle wave: 0->1->0->1... with period 20 patches."""
        cycle_pos = (patch_idx % 20) / 10.0  # 0..2
        if cycle_pos <= 1.0:
            val = cycle_pos  # Rising: 0 -> 1
        else:
            val = 2.0 - cycle_pos  # Falling: 1 -> 0
        return val


def generate_synthetic_csv(output_path: str, n_points: int = 10000, seed: int = 2021):
    """Generate synthetic dataset and save as CSV.

    Args:
        output_path: Path to save the CSV file
        n_points: Number of data points to generate
        seed: Random seed for data generation
    """
    config = get_synthetic_config()
    config.seed = seed
    generator = SyntheticDataGenerator(config)
    data = generator.generate(n_points)

    columns = [f'var_{i}' for i in range(config.n_source_variates)] + ['target']
    df = pd.DataFrame(data, columns=columns)

    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='h')
    df.insert(0, 'date', dates)

    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    return df


if __name__ == "__main__":
    import os
    output_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'Synthetic', 'synthetic.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generate_synthetic_csv(output_path)
