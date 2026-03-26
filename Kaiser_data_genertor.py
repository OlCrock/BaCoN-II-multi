from pyexpat import model
import numpy as np
import MGrowth as mg
import HMcode2020Emu as hmcodeemu
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA

class DataGenerator:

    def __init__(self, cosmo, kmin=1e-2, kmax=0.1, Nk=500):
        """
        cosmo: dictictonary:
            Omega_m, h, w0, wa,
            omega_b, omega_cdm,
            As, ns, Mnu, log10TAGN,
            z, b
        """
        self.cosmo = cosmo
        self.kmin = kmin
        self.kmax = kmax
        self.Nk = Nk

    # MGrowth parameters
    def background(self):
        a = 1 / (1 + self.cosmo['z'])
        return {
            'Omega_m': self.cosmo['Omega_m'],
            'h': self.cosmo['h'],
            'w0': self.cosmo['w0'],
            'wa': self.cosmo['wa'],
            'a_arr': np.array([a])
        }

    # HMcode parameters
    def hmcode_params(self):
        return {
            'omega_cdm':     [self.cosmo['omega_cdm']],
            'omega_baryon':  [self.cosmo['omega_b']],
            'As':            [self.cosmo['As']],
            'ns':            [self.cosmo['ns']],
            'hubble':        [self.cosmo['h']],
            'neutrino_mass': [self.cosmo['Mnu']],
            'w0':            [self.cosmo['w0']],
            'wa':            [self.cosmo['wa']],
            'log10TAGN':     [self.cosmo['log10TAGN']],
            'z':             [self.cosmo['z']],
            'omegarc':            [self.cosmo['omegarc']]
        }

    # Compute linear P(k) from HMcode
    def compute_pk(self):
        emulator = hmcodeemu.Matter_powerspectrum()
        k, pk_lin_total = emulator.get_linear_pk(nonu=False, **self.hmcode_params())
        return np.squeeze(k), np.squeeze(pk_lin_total)

    # Interpolate to fixed k-grid
    def interpolate_pk(self, k, Pm):
        mask = (k >= self.kmin) & (k <= self.kmax)
        k_lin = k[mask]
        P_lin = Pm[mask]

        k_target = np.logspace(np.log10(k_lin.min()), np.log10(k_lin.max()), self.Nk)
        P_target = np.interp(k_target, k_lin, P_lin)
        return k_target, P_target

    def growth(self, model, k=None):

        model_map = {
            'LCDM': mg.LCDM,
            'nDGP': mg.nDGP,
            'wCDM': mg.wCDM,
            'fR': mg.fR_HS
        }

        if model not in model_map:
            raise ValueError(f"Unknown model '{model}'")

        bg = self.background()

        # Pass fR0 only for f(R)
        if model == 'fR':
            bg['fR0'] = self.cosmo['fR0']

        cosmo_class = model_map[model]
        cosmo = cosmo_class(bg)

        if model == 'fR':
            print("Warning: k not provided for f(R), using default k-grid for growth parameters")
            if k is None:
                
                raise ValueError("k must be provided for f(R)")
            D, f = cosmo.growth_parameters(k, self.cosmo['fR0'])
        
        elif model == 'nDGP':
            D, f = cosmo.growth_parameters(self.cosmo['omegarc'])

        else:
            D, f = cosmo.growth_parameters()

        return D, f

    
    
    def kaiser(self, k, Pm, model, include_plin=False):
        b = self.cosmo['b']
        fR0_col = np.full(len(k), self.cosmo.get('fR0', 0.0))
        omegarc_col = np.full(len(k), self.cosmo.get('omegarc', 0.0))

        # LCDM reference
        D_LCDM, _ = self.growth('LCDM')

        if model == 'fR':
            # f(R) growth at original k
            k_native = np.logspace(np.log10(self.kmin), np.log10(self.kmax), len(Pm))  # same length as Pm
            D_fR, f_fR = self.growth('fR', k=k_native)
            D_fR = np.squeeze(D_fR)
            f_fR = np.squeeze(f_fR)

            # Interpolate D_fR and f_fR onto the final k-grid
            D_fR_interp = np.interp(k, k_native, D_fR)
            f_fR_interp = np.interp(k, k_native, f_fR)

            # Rescale Pm mode-by-mode
            P0 = (b**2 + 2/3*b*f_fR_interp + 1/5*f_fR_interp**2) * Pm * (D_fR_interp / D_LCDM)**2
            P2 = (4/3*b*f_fR_interp + 4/7*f_fR_interp**2) * Pm * (D_fR_interp / D_LCDM)**2
            P4 = (8/35*f_fR_interp**2) * Pm * (D_fR_interp / D_LCDM)**2

        else:
            # LCDM / nDGP / wCDM
            D, f = self.growth(model)
            if model != 'LCDM':
                Pm = Pm * (D / D_LCDM)**2
            P0 = (b**2 + 2/3*b*f + 1/5*f**2) * Pm
            P2 = (4/3*b*f + 4/7*f**2) * Pm
            P4 = (8/35*f**2) * Pm

        if include_plin:
            return np.column_stack((k, P0, P2, Pm))  # k, P0, P2, P4, Plin, ref
        else:
            return np.column_stack((k, P0, P2, P4))  # k, P0, P2, P4, ref
    





    # 7. Full pipeline
    def compute(self, model, include_plin=False):
        k, Pm = self.compute_pk()
        k_interp, P_interp = self.interpolate_pk(k, Pm)
        return self.kaiser(k_interp, P_interp, model, include_plin=include_plin)




def sampler(mean, std, low = None, high = None): 
    while True: 
        x = np.random.normal(mean, std) 
        if low is not None and high is not None:
            if low <= x <= high: 
                return x
        else:
            return x
        
def sample_w0_wa(mean_w0, std_w0, mean_wa, std_wa, low_w0=-3, high_w0=-0.3, low_wa=-3, high_wa=3):
    """
    Resample w0 and wa until the condition w0 + wa < 0 is satisfied.
    """
    while True:
        w0 = sampler(mean_w0, std_w0, low_w0, high_w0)
        wa = sampler(mean_wa, std_wa, low_wa, high_wa)
        if w0 + wa < 0:
            return w0, wa
        




# Planck cosmology

z_fid = 0.1 
b_fid = 1  # midpoint of [1.5, 3]

planck_params = {
    'Omega_m': 0.315,
    'h': 0.674,
    'w0': -1.0,
    'wa': 0.0,
    'omega_b': 0.05,
    'omega_cdm': 0.315 - 0.05, 
    'As': np.exp(3.07)*1.e-10,
    'ns': 0.96,
    'Mnu': 0.0,
    'log10TAGN': 7.8,
    'z': z_fid,
    'b': b_fid,
    'omegarc': 1e-1,    
    'fR0': 1e-7 # effectively GR

    
    }



planck_model = DataGenerator(planck_params)
P_planck = planck_model.compute(model='LCDM', include_plin=True)  

#Save Planck fiducial model
np.savetxt("Planck_fiducial.txt", P_planck)



# Generate and save 30 iterations for each model
# Also collect fR results to visualize how fR0 variation affects spectra
fR_results_list = []
fR0_values_list = []
LCDM_results_for_comparison = None

for iteration in range(1, 2):


    parameters_fr = {
    'Omega_m': 0.315,
    'h': 0.674,
    'w0': -1.0,
    'wa': 0.0,
    'omega_b': 0.05,
    'omega_cdm': 0.315 - 0.05, 
    'As': np.exp(3.07)*1.e-10,
    'ns': 0.96,
    'Mnu': 0.0,
    'log10TAGN': 7.8,
    'z': z_fid,
    'b': b_fid,
    'omegarc': 1e-1,    
    'fR0': np.random.uniform(1e-5, 1e-4) 
    }



   
    omega_m = sampler(0.3158, 0.009)

    w0, wa = sample_w0_wa(-1.0, 0.097, 0.0, 0.32)

    Parameters = {
    'Omega_m': omega_m,
    'h': 0.674,
    'w0': w0,
    'wa': wa,
    'omega_b': 0.05,
    'omega_cdm': omega_m - 0.05, 
    'As': sampler(2.199e-9, 2.199e-11),
    'ns': sampler(0.966, 0.007, 0.6, 1.2),
    'Mnu': 0.0,
    'log10TAGN': 7.8,
    'z': z_fid,
    'b': b_fid,
    
    'omegarc': sampler(1e-10, 0.173, 1e-12, 1),
    'fR0': 10**np.random.uniform(-5, -4)  # Uniform in log space from 1e-5 to 1e-4
    }
    
    

    
   
    model = DataGenerator(planck_params)
    results = {m: model.compute(model=m) for m in ['LCDM','nDGP','fR', 'wCDM']}
    """
    model_fr = DataGenerator(parameters_fr)
    results_fr = {m: model_fr.compute(model=m) for m in ['fR', 'LCDM']} 
    
    # Store fR results for plotting individual iterations
    fR_results_list.append(results_fr['fR'])
    fR0_values_list.append(parameters_fr['fR0'])
    if LCDM_results_for_comparison is None:
        LCDM_results_for_comparison = results_fr['LCDM']
    """

    
    # Create directories if they don't exist
    for m in ['LCDM', 'nDGP','fR', 'wCDM']:
        os.makedirs(m, exist_ok=True)
        np.savetxt(f"{m}/{iteration}.txt", results[m])
    
    print(f"Saved iteration {iteration}/1000, fR0={parameters_fr['fR0']:.2e}")
  
    

dg = DataGenerator(planck_params)

# Load all varying parameter results and plot them
def load_model_iterations(model_name, max_iterations=1000):
    """Load all saved iterations for a model"""
    data_dict = {}
    for i in range(1, max_iterations + 1):
        filepath = f"{model_name}/{i}.txt"
        if os.path.exists(filepath):
            data_dict[i] = np.loadtxt(filepath)
        else:
            break
    return data_dict

# Load all iterations
LCDM_iterations = load_model_iterations('LCDM')
nDGP_iterations = load_model_iterations('nDGP')
fR_iterations = load_model_iterations('fR')
wCDM_iterations = load_model_iterations('wCDM')

# Extract first iteration for reference k values
spec_LCDM = list(LCDM_iterations.values())[0] if LCDM_iterations else dg.compute('LCDM')
spec_nDGP = list(nDGP_iterations.values())[0] if nDGP_iterations else dg.compute('nDGP')
spec_fR = list(fR_iterations.values())[0] if fR_iterations else dg.compute('fR')
spec_wCDM = list(wCDM_iterations.values())[0] if wCDM_iterations else dg.compute('wCDM')

# Unpack (format: k, p0, p2, p4, plus any additional columns)
k = spec_LCDM[:, 0]
P0_LCDM, P2_LCDM, P4_LCDM = spec_LCDM[:, 1], spec_LCDM[:, 2], spec_LCDM[:, 3]
P0_nDGP, P2_nDGP, P4_nDGP = spec_nDGP[:, 1], spec_nDGP[:, 2], spec_nDGP[:, 3]
P0_fR, P2_fR, P4_fR = spec_fR[:, 1], spec_fR[:, 2], spec_fR[:, 3]
P0_wCDM, P2_wCDM, P4_wCDM = spec_wCDM[:, 1], spec_wCDM[:, 2], spec_wCDM[:, 3]


plt.style.use('dark_background')

models_dict = {
    'LCDM': LCDM_iterations,
    'wCDM': wCDM_iterations,
    'nDGP': nDGP_iterations,
    'f(R)': fR_iterations
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = ['P0', 'P2', 'P4']

# Load Planck fiducial for normalization
planck_spec = np.loadtxt('Planck_fiducial.txt')
P0_planck, P2_planck, P4_planck = planck_spec[:, 1], planck_spec[:, 2], planck_spec[:, 3]

for model_name, iterations_dict in models_dict.items():
    fig, ax = plt.subplots(figsize=(8,5))

    # Fonts
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10
    })

    # Calculate mean across all iterations
    P0_list, P2_list, P4_list = [], [], []
    k_iter = None
    
    for iteration_num, spec in iterations_dict.items():
        k_temp = spec[:, 0]
        P0, P2, P4 = spec[:, 1], spec[:, 2], spec[:, 3]
        if k_iter is None:
            k_iter = k_temp
        P0_list.append(P0)
        P2_list.append(P2)
        P4_list.append(P4)
    
    # Calculate means
    P0_mean = np.mean(P0_list, axis=0)
    P2_mean = np.mean(P2_list, axis=0)
    P4_mean = np.mean(P4_list, axis=0)
    
    # Compute inverse-normalized spectra relative to Planck
    y0_mean = (P0_mean / P0_planck)**-1
    y2_mean = (P2_mean / P2_planck)**-1
    y4_mean = (P4_mean / P4_planck)**-1
    
    # Plot mean lines
    ax.semilogx(k_iter, y0_mean, label='P0', color=colors[0], linewidth=2)
    ax.semilogx(k_iter, y2_mean, label='P2', color=colors[1], linewidth=2)
    ax.semilogx(k_iter, y4_mean, label='P4', color=colors[2], linewidth=2)

    # Axis lines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('white')
        spine.set_linewidth(1.2)

    # Labels and legend
    ax.set_xlabel(r'$k\,[h/\mathrm{Mpc}]$')
    ax.set_ylabel(r'$(P_\ell / P_0)^{-1}$')
    ax.set_title(f'{model_name} Multipoles')
    ax.legend(frameon=False, loc='lower left')

    plt.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# Setup plotting
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use('dark_background')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
})

# Build data structures
LCDM_P0_arr = np.array([v[:, 1] for v in LCDM_iterations.values()])
LCDM_P2_arr = np.array([v[:, 2] for v in LCDM_iterations.values()])
LCDM_P4_arr = np.array([v[:, 3] for v in LCDM_iterations.values()])

LCDM_P0_std = np.std(LCDM_P0_arr, axis=0)
LCDM_P2_std = np.std(LCDM_P2_arr, axis=0)
LCDM_P4_std = np.std(LCDM_P4_arr, axis=0)

P0_planck = LCDM_results_for_comparison[:, 1]
P2_planck = LCDM_results_for_comparison[:, 2]
P4_planck = LCDM_results_for_comparison[:, 3]

k_ref = fR_results_list[0][:, 0]
fR0_arr = np.array(fR0_values_list)

print(f"[DEBUG] k_ref shape: {k_ref.shape}")
print(f"[DEBUG] fR0 range: [{fR0_arr.min():.3e}, {fR0_arr.max():.3e}]")
print(f"[DEBUG] LCDM_P0_std range: [{LCDM_P0_std.min():.3e}, {LCDM_P0_std.max():.3e}]")

# Small histogram of sampled fR0 values (log-space)
try:
    import matplotlib.pyplot as _plt
    _fig = _plt.figure(figsize=(8, 4), facecolor='#0a0a0a')
    _ax = _fig.add_subplot(111, facecolor='#151515')
    log10_fR0 = np.log10(fR0_arr)
    _ax.hist(log10_fR0, bins=40, color='#4FC3F7', edgecolor='white', linewidth=0.5, alpha=0.85)
    _ax.set_xlabel(r'$\log_{10}(f_{R0})$', fontsize=11, color='white')
    _ax.set_ylabel('Counts', fontsize=11, color='white')
    _ax.set_title(f'Sampled fR0 Distribution (N={len(fR0_arr)})', fontsize=12, color='white')
    _ax.grid(alpha=0.2, color='gray')
    _ax.tick_params(colors='white')
    _plt.tight_layout()
    _plt.show()
    _plt.close(_fig)
except Exception as _e:
    print(f'Could not display fR0 histogram: {_e}')
# Colormap setup
norm_col = LogNorm(vmin=fR0_arr.min(), vmax=fR0_arr.max())
cmap = plt.cm.plasma

# ─────────────────────────────────────────────────────────────────────────────
# Create figure
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 11), facecolor='#0a0a0a')
gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.4, height_ratios=[1, 0.9, 1])

axes_clean = [fig.add_subplot(gs[0, c], facecolor='#151515') for c in range(3)]
axes_scatter = [fig.add_subplot(gs[1, c], facecolor='#151515') for c in range(3)]
ax_snr = fig.add_subplot(gs[2, :], facecolor='#151515')

multipole_config = [
    ('P0', 1, P0_planck, LCDM_P0_std, '#4FC3F7'),
    ('P2', 2, P2_planck, LCDM_P2_std, '#FFB74D'),
    ('P4', 3, P4_planck, LCDM_P4_std, '#81C784'),
]

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Clean ratio plots
# ─────────────────────────────────────────────────────────────────────────────

print("\n[PLOT] Rendering Part 1: fR/ΛCDM ratios...")
for col, (name, col_idx, planck_ref, lcdm_std, color) in enumerate(multipole_config):
    ax = axes_clean[col]
    
    for fR_spec, fR0_val in zip(fR_results_list, fR0_values_list):
        ratio = fR_spec[:, col_idx] / planck_ref
        line_color = cmap(norm_col(fR0_val))
        ax.semilogx(k_ref, ratio, alpha=0.5, color=line_color, linewidth=1.2)
    
    ax.axhline(1, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel(rf'$P_\ell^{{f(R)}} / P_\ell^{{\Lambda\mathrm{{CDM}}}}$')
    ax.set_title(f'{name}  —  fR/ΛCDM ratio', color=color)
    ax.grid(True, alpha=0.15, color='gray')
    ax.set_ylim([0.90, 1.10])

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: ΛCDM noise floor (log-log scale)
# ─────────────────────────────────────────────────────────────────────────────

print("[PLOT] Rendering Part 2: ΛCDM noise floor...")
for col, (name, col_idx, planck_ref, lcdm_std, color) in enumerate(multipole_config):
    ax = axes_scatter[col]
    
    frac_scatter = lcdm_std / planck_ref
    
    # Use log-log to better show the noise structure
    ax.loglog(k_ref, frac_scatter, color=color, linewidth=2.5)
    ax.fill_between(k_ref, frac_scatter/3, frac_scatter*3, 
                     alpha=0.2, color=color, label='±3× noise')
    
    ax.set_ylabel('Fractional noise (σ/P)')
    ax.set_title(f'{name}  —  ΛCDM parameter noise', color=color)
    ax.grid(True, alpha=0.15, color='gray', which='both')
    ax.legend(frameon=False, fontsize=8)

axes_scatter[1].set_xlabel(r'$k\,[h/\mathrm{Mpc}]$')

# ─────────────────────────────────────────────────────────────────────────────
# Part 3: SNR curves
# ─────────────────────────────────────────────────────────────────────────────

print("[PLOT] Rendering Part 3: SNR curves...")
snr_curves = {name: [] for name, *_ in multipole_config}

for fR_spec, fR0_val in zip(fR_results_list, fR0_values_list):
    for name, col_idx, planck_ref, lcdm_std, color in multipole_config:
        ratio = fR_spec[:, col_idx] / planck_ref
        signal = np.max(np.abs(ratio - 1.0))
        noise = np.mean(lcdm_std / planck_ref)
        snr = signal / noise if noise > 0 else 0.0
        snr_curves[name].append((fR0_val, snr))

threshold_fR0 = {}
for name, _, _, _, color in multipole_config:
    pts = sorted(snr_curves[name], key=lambda x: x[0])
    fR0s = [p[0] for p in pts]
    snrs = [p[1] for p in pts]
    
    ax_snr.loglog(fR0s, snrs, 'o-', color=color, label=name,
                  linewidth=2.2, markersize=6, alpha=0.85)
    
    # Find SNR=1 crossing
    for i in range(len(pts) - 1):
        if pts[i][1] < 1.0 <= pts[i+1][1]:
            threshold_fR0[name] = pts[i+1][0]
            break

ax_snr.axhline(1.0, color='white', linestyle='--', linewidth=1, alpha=0.7, label='SNR = 1')
ax_snr.axhline(3.0, color='white', linestyle=':', linewidth=0.8, alpha=0.5, label='SNR = 3')
ax_snr.set_xlabel(r'$f_{R0}$')
ax_snr.set_ylabel('SNR')
ax_snr.set_title(r'Signal-to-noise ratio vs $f_{R0}$')
ax_snr.legend(frameon=False, ncol=2, loc='best', fontsize=8)
ax_snr.grid(True, alpha=0.15, color='gray', which='both')

# Mark thresholds
for name, threshold in threshold_fR0.items():
    color = [c for n, _, _, _, c in multipole_config if n == name][0]
    ax_snr.axvline(threshold, color=color, linestyle='-.', linewidth=0.9, alpha=0.5)

fig.suptitle(r'$f(R)$ Diagnostic  —  Classifiable Signal vs Noise', fontsize=12)

plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  f(R) GRAVITY DETECTION THRESHOLDS")
print("="*70)
for name in ['P0', 'P2', 'P4']:
    if name in threshold_fR0:
        print(f"  {name}: fR0 ≈ {threshold_fR0[name]:.3e}")
    else:
        print(f"  {name}: No SNR=1 crossing found")
print("="*70 + "\n")

