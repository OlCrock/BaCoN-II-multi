# Summary of Changes to BaCoN-II

## Overview

The following changes extend BaCoN to work with **Kaiser multipole power spectra** (P0, P2, P4) instead of the original matter power spectrum, and implement a **linear Kaiser noise covariance model** appropriate for redshift-space multipoles.

---

## 1. Data Generation 

## — `Kaiser_data_genertor.py` (new file)

Generates training data using the Kaiser multipole decomposition:
- Uses **HMcode2020Emu** for the linear matter power spectrum P(k)
- Uses **MGrowth** to compute growth factors D and growth rates f for each cosmological model (LCDM, wCDM, nDGP, f(R))
- Applies the **Kaiser formula** to compute multipoles:
  - P0 = (b² + 2/3 bf + 1/5 f²) × Pm
  - P2 = (4/3 bf + 4/7 f²) × Pm
  - P4 = (8/35 f²) × Pm
- Rescales spectra relative to LCDM via (D/D_LCDM)²
- For f(R): growth rate f is k-dependent (interpolated onto the output k-grid)
- Stores output as columns: **k, P0, P2, P4**
- Generates 1000 uniquely sampled cosmologies per model, varying Ωm, As, ns, w0, wa, Ωrc, fR0
- Saves a **Planck fiducial** spectrum used for normalisation in the format 
**k, P0, P2, P4, f** (where f is the growth factor utilised to compute errors)

## — `NonLinear_data_genertor.py` (new file)

Generates training data using an ad hoc non-linear multipole decomposition:
- Uses **HMcode2020Emu** for the non-linear matter power spectrum P(k)
- Uses **MGEmu** to compute boost for each cosmological model (LCDM, wCDM, nDGP, f(R))
- Solves the **Legendre projection** integral numerically to compute multipoles
- Rescales spectra relative to LCDM via the non-linear boost
- Generates 1000 uniquely sampled cosmologies per model, varying Ωm, As, ns, w0, wa, Ωrc, fR0
- Saves a **Planck fiducial** spectrum used for normalisation in the format 
**k, P0, P2, P4, f** (where f is the growth factor utilised to compute errors)



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

## 4. Training Parameters — `train-parameters.py`

- Added `noise_model` parameter
- Added Kaiser-specific parameters: `kaiser_bias`, `kaiser_ng`, `kaiser_veff`
- All passed through to `train.py` via subprocess

---

## 6. Backward Compatibility

- All new parameters have defaults that reproduce the original BaCoN behaviour
- `noise_model='default'` uses the original noise model exactly as before
- Old model log files without Kaiser parameters are handled gracefully via try/except blocks
- The data format is unchanged for `noise_model='default'` (no f column expected)

