# Summary of Changes to BaCoN-II

## Overview

The following changes extend BaCoN to work with **Kaiser multipole power spectra** (P0, P2, P4) instead of the original matter power spectrum, and implement a **linear Kaiser noise covariance model** appropriate for redshift-space multipoles.

---

## 1. Data Generation — `Kaiser_data_genertor.py` (new file)

Generates training data using the Kaiser multipole decomposition:
- Uses **HMcode2020Emu** for the linear matter power spectrum P(k)
- Uses **MGrowth** to compute growth factors D and growth rates f for each cosmological model (LCDM, wCDM, nDGP, f(R))
- Applies the **Kaiser formula** to compute multipoles:
  - P0 = (b² + 2/3 bf + 1/5 f²) × Pm
  - P2 = (4/3 bf + 4/7 f²) × Pm
  - P4 = (8/35 f²) × Pm
- Rescales spectra relative to LCDM via (D/D_LCDM)²
- For f(R): growth rate f is k-dependent (interpolated onto the output k-grid)
- Stores output as columns: **k, P0, P2, P4, f** (growth rate stored as last column for use by the noise model)
- Generates 1000 uniquely sampled cosmologies per model, varying Ωm, As, ns, w0, wa, Ωrc, fR0
- Saves a **Planck fiducial** spectrum used for normalisation

---

## 2. Noise Model — `data_generator.py`

### `generate_noise()` function
- Added `noise_model` parameter: `'default'` or `'linear_kaiser'`
- Added parameters: `growth_f`, `bias`, `ng`, `veff`
- **`'default'`**: Original BaCoN noise model (cosmic variance + shot noise + systematics), unchanged
- **`'linear_kaiser'`**: Full Kaiser covariance for monopole and quadrupole:
  - **Cov(0,0)**: Monopole covariance with prefactors involving β = f/b up to β⁴
  - **Cov(2,2)**: Quadrupole covariance with corresponding prefactors
  - Shot noise is included within the Kaiser covariance terms
  - Systematic errors can still be added on top via `add_sys`

### `DataGenerator` class
- Added instance variables: `noise_model`, `kaiser_bias`, `kaiser_ng`, `kaiser_veff`
- When `noise_model='linear_kaiser'`:
  - Extracts growth rate f from the **last column** of each data file
  - Removes the f column before processing the spectrum
  - Passes f, bias, ng, veff to `generate_noise()` for both cosmic variance and shot noise calls

### `create_generators()` function
- Reads `noise_model`, `kaiser_bias`, `kaiser_ng`, `kaiser_veff` from FLAGS
- Backward compatible: defaults to `'default'` if not found in older model logs

---

## 3. Training — `train.py`

### New command-line arguments
- `--noise_model` (default `'default'`)
- `--kaiser_bias` (default 1.0)
- `--kaiser_ng` (default None)
- `--kaiser_veff` (default 0.28, z=0.1)

### Post-training P0 diagnostic plot
After training completes, generates one plot per model class showing:
- **Bold solid line**: Mean normalised P0 across validation samples (noise averages out ≈ non-noisy mean)
- **Dashed lines**: 5 random noisy realisations from the validation set overlaid
- Normalisation: the data generator stores X = P/P_fid − 1, so the plot transforms to (X + 1)⁻¹ = (P/P_fid)⁻¹ to match the Kaiser_data_generator convention (values close to 1)
- Dark background style matching Kaiser_data_generator plots
- Saved to the model output directory as `P0_{model_name}.png`

---

## 4. Training Parameters — `train-parameters.py`

- Added `noise_model` parameter
- Added Kaiser-specific parameters: `kaiser_bias`, `kaiser_ng`, `kaiser_veff`
- All passed through to `train.py` via subprocess

---

## 5. Testing — `test.py`

- Added `--noise_model` argument (can override the training model's setting)
- Added Kaiser parameter arguments for consistency

---

## 6. Backward Compatibility

- All new parameters have defaults that reproduce the original BaCoN behaviour
- `noise_model='default'` uses the original noise model exactly as before
- Old model log files without Kaiser parameters are handled gracefully via try/except blocks
- The data format is unchanged for `noise_model='default'` (no f column expected)

