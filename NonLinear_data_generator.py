from pyexpat import model
import numpy as np
import MGrowth as mg
import HMcode2020Emu as hmcodeemu
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA
from scipy.integrate import simpson
import MGEmu as mgemu



class DataGenerator:

    def __init__(self, cosmo, kmin=1e-2, kmax=0.2, Nk=500, N_mu=2000):
        """
        cosmo: dictionary containing:
            Omega_m, h, w0, wa,
            omega_b, omega_cdm,
            As, ns, Mnu, log10TAGN,
            z, b, omegarc, fR0,
            sigma_p   ← nuisance parameter (Mpc/h), sampled not computed
        """
        self.cosmo = cosmo
        self.kmin  = kmin
        self.kmax  = kmax
        self.Nk    = Nk
        self.N_mu  = N_mu
        self._emulator = None



    @property
    def emulator(self):
        if self._emulator is None:
            self._emulator = hmcodeemu.Matter_powerspectrum()
        return self._emulator


    def background(self):
        a = 1 / (1 + self.cosmo['z'])
        return {
            'Omega_m': self.cosmo['Omega_m'],
            'h':       self.cosmo['h'],
            'w0':      self.cosmo['w0'],
            'wa':      self.cosmo['wa'],
            'a_arr':   np.array([a])
        }

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
        }
    
    def MG_growth_params(self):
        return{
            'Omega_m':  [self.cosmo['Omega_m']],
            'Omega_b':  [self.cosmo['omega_b']],
            'ns':  [self.cosmo['ns']],
            'H0':  [self.cosmo['h']*100],
            'Omega_nu' :  [(self.cosmo['Mnu']) / (93.14 * (self.cosmo['h']**2))],  # Convert Mnu to Omega_nu
            'As':  [self.cosmo['As']],
            'w0':  [self.cosmo['w0']],
            'wa':  [self.cosmo['wa']],
            'xi':  [0],
            'fR0':  [self.cosmo['fR0']],
            'omegarc':  [self.cosmo['omegarc']],
            'z':  [self.cosmo['z']]
        }

    # ------------------------------------------------------------------ #
    #  Power spectra from emulator                                         #
    # ------------------------------------------------------------------ #

    def compute_pk_linear(self):
        k, pk = self.emulator.get_linear_pk(nonu=False, **self.hmcode_params())
        return np.squeeze(k), np.squeeze(pk)

    def compute_pk_nonlinear(self, baryonic_boost=True):
        k, pk = self.emulator.get_nonlinear_pk(
            baryonic_boost=baryonic_boost, **self.hmcode_params()
        )
        return np.squeeze(k), np.squeeze(pk)

    def interpolate_pk(self, k, Pm):
        """Interpolate P(k) onto a fixed log-spaced k grid between kmin and kmax."""
        mask  = (k >= self.kmin) & (k <= self.kmax)
        k_out = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.Nk)
        P_out = np.interp(k_out, k[mask], Pm[mask])
        return k_out, P_out

    # ------------------------------------------------------------------ #
    #  Non-linear boost (MGEmu emulator)                                  #
    # ------------------------------------------------------------------ #

    def boost(self, model):
        """
        use the MGEmu emulator to compute the boost for all non LCDM mdoels 
        """
        
        params = self.MG_growth_params()

        if model == 'fR':
            emulator = mgemu.MG_boost(model='fR')
            kvals, boost_nl = emulator.get_nonlinear_boost(**params)

        elif model == 'nDGP':
            emulator = mgemu.MG_boost(model='DGP')
            kvals, boost_nl = emulator.get_nonlinear_boost(**params)

        elif model == 'wCDM':
            emulator = mgemu.MG_boost(model='ds')
            kvals, boost_nl = emulator.get_nonlinear_boost(**params)

        return np.squeeze(kvals), np.squeeze(boost_nl)

    # ------------------------------------------------------------------ #
    #  Growth factors via MGrowth                                          #
    # ------------------------------------------------------------------ #

    def growth(self, model, k=None, omegarc=None, fR0=None):
        model_map = {
            'LCDM': mg.LCDM,
            'nDGP': mg.nDGP,
            'wCDM': mg.wCDM,
            'fR':   mg.fR_HS
        }
        if model not in model_map:
            raise ValueError(f"Unknown model '{model}'")

        bg = self.background()

        cosmo_obj = model_map[model](bg)

        if model == 'fR':
            if fR0 is None:
                fR0 = self.cosmo['fR0']
            D, f = cosmo_obj.growth_parameters(k, fR0)
        elif model == 'nDGP':
            if omegarc is None:
                omegarc = self.cosmo['omegarc']
            D, f = cosmo_obj.growth_parameters(omegarc)
        else:
            D, f = cosmo_obj.growth_parameters()

        return np.squeeze(D), np.squeeze(f)

    # ------------------------------------------------------------------ #
    #  Core: numerical mu integration → P0, P2, P4                        #
    #                                                                      #
    #  Single method handles both linear and non-linear.                  #
    #  sigma_p is READ from cosmo dict — sampled nuisance parameter.      #
    #  sigma_v is COMPUTED from linear Pk for BAO damping scale.          #
    # ------------------------------------------------------------------ #

    def _compute_multipoles(self, k, Pk_lin, Pk_nl, model, include_plin=False, Pk_lin_planck=None, sigp2_ref=None):
        """
        Numerically integrate P(k, mu) over mu to obtain P0, P2.

        P(k, mu) = (b + f(k)*mu^2)^2 * P_nl(k) / (1 + k^2*mu^2*sigma_p^2)

        where:
            P_nl(k)     = HMcode non-linear power spectrum (with MG boost)
            sigma_p     = sampled nuisance parameter from cosmo dict (Finger-of-God damping)
            f(k)        = growth rate from MGrowth
            Pk_lin_planck = reference Planck linear spectrum (unused, kept for compatibility)
            sigp2_ref = pre-computed reference sigma_p^2 from Planck (optional)
        """

        b   = self.cosmo['b']
        sigmap2 = self.cosmo['sigma_p'] ** 2  # sampled

        # --- Growth rate f(k) ---
        D_LCDM, f_LCDM = self.growth('LCDM')

        if model == 'fR':
            D_mg, f_mg = self.growth('fR', k=k, fR0=self.cosmo['fR0'])
            f_mg = np.atleast_1d(f_mg) * np.ones_like(k)

        elif model == 'nDGP':
            D_mg, f_mg = self.growth('nDGP', omegarc=self.cosmo['omegarc'])
            f_mg = np.atleast_1d(f_mg) * np.ones_like(k)
     
        elif model == 'wCDM':
            D_mg, f_mg = self.growth('wCDM')
            f_mg = np.atleast_1d(f_mg) * np.ones_like(k)
        
        else:   # LCDM
            D_mg, f_mg = self.growth('LCDM')
            f_mg = np.atleast_1d(f_mg) * np.ones_like(k)
    

        # --- Rescale Pk_nl by non-linear boost (includes growth ratio factor) ---
        if model != 'LCDM':
            k_boost, boost_nl = self.boost(model)
            Xi = np.interp(k, k_boost, boost_nl)
            Pk_nl = Pk_nl * Xi

        # --- Build mu grid, much faster for numeric integration than looping ---
        mu_arr = np.linspace(-1, 1, self.N_mu)   # (N_mu,)

        # --- Build 2D (Nk, N_mu) arrays ---
        k_arr  = k[:, None]                         # (Nk, 1)
        mu2d   = mu_arr[None, :]                    # (1, N_mu)
        f_arr  = f_mg[:, None]                      # (Nk, 1)
        Pnl2d  = Pk_nl[:, None]                     # (Nk, 1)

        # Use non-linear HMcode power spectrum directly as the de-wiggled spectrum
        Pkmu = ((b + f_arr * mu2d**2)**2) / (1.0 + k_arr**2 * mu2d**2 * sigmap2) * Pnl2d

        # --- Legendre polynomials ---
        L0 = np.ones_like(mu2d)                                      # ell=0
        L2 = 0.5   * (3.0*mu2d**2 - 1.0)                             # ell=2
        L4 = 0.125 * (35.0*mu2d**4 - 30.0*mu2d**2 + 3.0)            # ell=4

        # --- Project onto multipoles ---
        P0 = (1.0/2.0) * simpson(Pkmu * L0, x=mu_arr, axis=1)        # (Nk,)
        P2 = (5.0/2.0) * simpson(Pkmu * L2, x=mu_arr, axis=1)
        P4 = (9.0/2.0) * simpson(Pkmu * L4, x=mu_arr, axis=1)

        # calculate reference sigma_p^2 from Planck linear spectrum
        if Pk_lin_planck is None:
            Pk_lin_planck = Pk_lin
        
        if sigp2_ref is None:
            sigp2_ref = f_LCDM/(6 * np.pi**2) * simpson(Pk_lin_planck, x=k)

        if model == 'LCDM':
            print(f"Reference sigma_p^2 (from Planck): {sigp2_ref}")

        if include_plin:
            return np.column_stack((k, P0, P2, Pk_lin))
        return np.column_stack((k, P0, P2))


    def compute(self, model, baryonic_boost=True, include_plin=False, Pk_lin_planck=None, sigp2_ref=None):
        """
        Compute non-linear multipoles P0 and P2.

        Parameters
        ----------
        model        : 'LCDM', 'wCDM', 'nDGP', 'fR'
        baryonic_boost: bool — include AGN baryonic feedback
        include_plin : bool — append linear Pk as final column
        Pk_lin_planck : array, optional — reference Planck linear Pk (unused, kept for compatibility)
        sigp2_ref : float, optional — pre-computed reference sigma_p^2 from Planck
        """

        # Get linear Pk for reference
        k_lin, Pk_lin = self.compute_pk_linear()
        k_interp, Plin_interp = self.interpolate_pk(k_lin, Pk_lin)

        # Get non-linear Pk
        k_nl, Pk_nl = self.compute_pk_nonlinear(baryonic_boost=baryonic_boost)
        _, Pnl_interp = self.interpolate_pk(k_nl, Pk_nl)

        return self._compute_multipoles(
            k_interp, Plin_interp, Pnl_interp, model, include_plin, Pk_lin_planck, sigp2_ref
        )


# ======================================================================== #
#  Sampling utilities                                                        #
# ======================================================================== #

def sampler(mean, std, low=None, high=None):
    """Sample from a truncated Gaussian."""
    while True:
        x = np.random.normal(mean, std)
        if low is not None and high is not None:
            if low <= x <= high:
                return x
        else:
            return x


def sample_w0_wa(mean_w0, std_w0, mean_wa, std_wa,
                 low_w0=-1.3, high_w0=-0.5, low_wa=-2, high_wa=0.5):
    """
    Sample w0 and wa jointly, enforcing w0 + wa < 0
    so that w(a) < 0 at all redshifts.
    """
    while True:
        w0 = sampler(mean_w0, std_w0, low_w0, high_w0)
        wa = sampler(mean_wa, std_wa, low_wa, high_wa)
        if w0 + wa < 0:
            return w0, wa


def plot_normalized_multipoles(results_dict, planck_spectrum,
                               iteration=None, save=False,
                               outdir="normalized_plots"):
    """
    Plot one figure per model with multipoles normalized to Planck:

        P0/P0_planck, P2/P2_planck

    Parameters
    ----------
    results_dict : dict
        Mapping model name -> array with columns [k, P0, P2]
    planck_spectrum : array
        Array with columns [k, P0, P2, ...]
    iteration : int or None
        Optional iteration label for figure titles / filenames
    save : bool
        Save figures if True
    outdir : str
        Output folder for saved figures
    """
    k_ref = planck_spectrum[:, 0]
    planck_multipoles = planck_spectrum[:, 1:3]

    labels = [r"$P_0/P_0^{\rm Planck}$", r"$P_2/P_2^{\rm Planck}$"]

    if save:
        os.makedirs(outdir, exist_ok=True)

    for model_name, spec in results_dict.items():
        k = spec[:, 0]
        multipoles = spec[:, 1:3]

        # Interpolate Planck multipoles if grids differ
        planck_interp = np.column_stack([
            np.interp(k, k_ref, planck_multipoles[:, 0]),
            np.interp(k, k_ref, planck_multipoles[:, 1]),
        ])

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = multipoles / planck_interp

        fig, ax = plt.subplots(figsize=(8, 5))
        plotted_any = False
        for ell in range(2):
            mask = np.isfinite(k) & (k > 0.0) & np.isfinite(ratio[:, ell])
            if np.any(mask):
                ax.plot(k[mask], ratio[mask, ell], lw=1.8, label=labels[ell])
                plotted_any = True

        if not plotted_any:
            plt.close(fig)
            print(f"Skipping normalized plot for {model_name}: no positive finite k values to show.")
            continue

        ax.axhline(1.0, ls='--', lw=1.0, color='black', alpha=0.7)
        ax.set_xscale('log')
        positive_k = k[np.isfinite(k) & (k > 0.0)]
        if positive_k.size > 0:
            ax.set_xlim(positive_k.min(), positive_k.max())
        ax.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
        ax.set_ylabel("Normalized multipole")

        if iteration is None:
            ax.set_title(f"{model_name}: multipoles normalized to Planck")
        else:
            ax.set_title(f"{model_name}: normalized multipoles (iteration {iteration})")

        ax.legend(frameon=False)
        plt.tight_layout()

        if save:
            if iteration is None:
                fname = f"{model_name}_normalized.png"
            else:
                fname = f"{model_name}_{iteration}_normalized.png"
            plt.savefig(os.path.join(outdir, fname), dpi=180, bbox_inches='tight')

        plt.show()


def plot_fR0_sensitivity(fR0_results_list, fR0_values_list, planck_spectrum):
    """
    Plot f(R) gravity spectra (P0 only) normalized to Planck for varying fR0 parameters.
    
    Parameters
    ----------
    fR0_results_list : list of arrays
        fR spectra computed with varying fR0 values
    fR0_values_list : list of floats
        Corresponding fR0 parameter values (from 1e-10 to 1e-6)
    planck_spectrum : array
        Planck LCDM spectrum with columns [k, P0, P2, ...]
    """
    if len(fR0_results_list) == 0:
        print("No fR0 sweep results to plot.")
        return
    
    k_ref = planck_spectrum[:, 0]
    P0_planck = planck_spectrum[:, 1]
    
    # Setup colormap for fR0 range [1e-10, 1e-4] - viridis
    fR0_arr = np.array(fR0_values_list)
    norm = LogNorm(vmin=1e-30, vmax=1e-7)
    cmap = plt.cm.viridis
    
    fig, (ax, cbar_ax) = plt.subplots(1, 2, figsize=(8, 8), facecolor='#0a0a0a',
                                       gridspec_kw={'width_ratios': [4, 0.3]})
    ax.set_facecolor('#000000')
    cbar_ax.set_facecolor('#0a0a0a')
    
    # Plot P0 ratios
    for fR_spec, fR0_val in zip(fR0_results_list, fR0_values_list):
        k = fR_spec[:, 0]
        P0_fR = fR_spec[:, 1]
        
        # Interpolate Planck to match fR k-grid
        P0_planck_interp = np.interp(k, k_ref, P0_planck)
        
        # Compute ratio
        ratio_P0 = P0_fR / P0_planck_interp
        
        # Color from fR0 value
        color = cmap(norm(fR0_val))
        
        ax.semilogx(k, ratio_P0, color=color, linewidth=2.0, alpha=0.8)
    
    # Reference line
    ax.axhline(1.0, color='white', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r'$k\,[h/\mathrm{Mpc}]$', fontsize=18, color='white')
    ax.set_ylabel(r'$P_0^{f(R)} / P_0^{\Lambda\mathrm{CDM}}$', fontsize=18, color='white')
    ax.set_title(r'Monopole: f(R) normalized to Planck', fontsize=19, color='white')
    ax.grid(True, alpha=0.35, color='gray')
    ax.tick_params(colors='white', labelsize=16)
    
    # Add white border/spines
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1.5)
    
    # Add colorbar as separate subplot
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'$f_{R0}$', fontsize=18, color='white')
    cbar.ax.tick_params(colors='white', labelsize=14)
    
    plt.tight_layout()
    plt.show()


# ======================================================================== #
#  Fiducial cosmology                                                        #
# ======================================================================== #

z_fid = 0.1
b_fid = 1.0

planck_params = {
    'Omega_m':    0.315,
    'h':          0.674,
    'w0':        -1.0,
    'wa':         0.0,
    'omega_b':    0.05,
    'omega_cdm':  0.315 - 0.05,
    'As':         np.exp(3.07) * 1.e-10,
    'ns':         0.96,
    'Mnu':        0.0,
    'log10TAGN':  7.8,
    'z':          z_fid,
    'b':          b_fid,
    'omegarc':    1e-1,
    'fR0':        1e-7,  # small fR0 for near-LCDM reference
    'sigma_p':    3.78,    # Mpc/h — fiducial nuisance value
}

planck_model = DataGenerator(planck_params)
P_planck = planck_model.compute(model='LCDM', include_plin=True)
np.savetxt("Planck_fiducial_nonlin.txt", P_planck)

# Extract Planck linear spectrum and growth factor for reference
k_planck = P_planck[:, 0]
Pk_lin_planck = P_planck[:, -1]  # Last column is linear Pk
D_planck, f_planck = planck_model.growth('LCDM')
sigp2_ref = f_planck/(6 * np.pi**2) * simpson(Pk_lin_planck, x=k_planck)
print(f"Planck reference sigma_p^2: {sigp2_ref}\n")

plot_normalized_figures = True

# ======================================================================== #
#  Prepare for fR0 parameter sweep plot                                     #
# ======================================================================== #

fR0_sweep_results = []
fR0_sweep_values = []

# ======================================================================== #
#  Generation loop                                                           #
# ======================================================================== #

for iteration in range(1, 201):

    omega_m = sampler(0.3158, 0.009, 0.2899, 0.3392)


    w0, wa = sample_w0_wa(-1.0, 0.097, 0.0, 0.32)


    Parameters = {
        'Omega_m': omega_m,
        'h': 0.674,
        'w0': w0,
        'wa': wa,
        'omega_b': 0.05,
        'omega_cdm': omega_m - 0.05,
        'As': sampler(2.199e-9, 2.199e-11, 1.7e-9, 2.5e-9),
        'ns': sampler(0.966, 0.007, 0.9432, 0.9862),
        'Mnu': 0.0,
        'log10TAGN': 7.8,
        'z': z_fid,
        'b': b_fid,
        
        'omegarc': sampler(1e-2, 0.173, 1e-3, 100),
        'fR0': np.random.uniform(1e-6, 1e-5),  # uniform in log space from 1e-10 to 1e-3


        'sigma_p': np.random.uniform(3.0, 6.0),   # sampled nuisance
        'Xi':         0.0,   
    }

    model = DataGenerator(Parameters)
    results = {m: model.compute(model=m, Pk_lin_planck=Pk_lin_planck, sigp2_ref=sigp2_ref)
               for m in ['LCDM', 'nDGP', 'fR']}

    for m in ['LCDM', 'nDGP', 'fR']:
        os.makedirs(m, exist_ok=True)
        np.savetxt(f"{m}/{iteration}.txt", results[m])

    # Collect fR results with Planck cosmology but varying fR0 for sensitivity plot
    if iteration == 1:
        planck_fR0_params = planck_params.copy()
        for fR0_val in np.logspace(-30, -7, 30):
            planck_fR0_params['fR0'] = fR0_val
            dg_sweep = DataGenerator(planck_fR0_params)
            result_fR = dg_sweep.compute(model='fR', Pk_lin_planck=Pk_lin_planck, sigp2_ref=sigp2_ref)
            fR0_sweep_results.append(result_fR)
            fR0_sweep_values.append(fR0_val)

    print(f"Saved iteration {iteration}, sigma_p={Parameters['sigma_p']:.2f}")

if plot_normalized_figures:
    # Plot fR0 parameter sweep if available
    if len(fR0_sweep_results) > 0:
        plot_fR0_sensitivity(fR0_sweep_results, fR0_sweep_values, P_planck)


# ======================================================================== #
#  Simpson Integration Convergence Test                                     #
# ======================================================================== #

def test_simpson_convergence(planck_params, model='LCDM', k_test_idx=100):
    """
    Test convergence of Simpson integration with respect to number of mu points.
    
    Parameters
    ----------
    planck_params : dict
        Cosmological parameters
    model : str
        Which model to test ('LCDM', 'wCDM', 'nDGP', 'fR')
    k_test_idx : int
        Which k index to test (default: middle of k array)
    """
    import matplotlib.pyplot as plt
    
    # Test with different N_mu values
    N_mu_values = [5,10, 12,15,17, 20,25, 30, 50, 100, 200, 300]
    P0_values = []
    P2_values = []
    
    for N_mu in N_mu_values:
        dg = DataGenerator(planck_params, N_mu=N_mu)
        P = dg.compute(model=model)
        P0 = P[k_test_idx, 1]
        P2 = P[k_test_idx, 2]
        
        P0_values.append(P0)
        P2_values.append(P2)
        
        print(f"{N_mu:>6} | {P0:>15.6e} | {P2:>15.6e}")
    
    # Plot convergence on separate graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # P0 plot
    ax1.plot(N_mu_values, P0_values, 'o-', linewidth=2.5, markersize=8, color='#1f77b4')
    ax1.set_xlabel('Number of μ integration points', fontsize=12)
    ax1.set_ylabel('P0 value', fontsize=12)
    ax1.set_title(f'{model}: P0 Convergence', fontsize=13)
    ax1.grid(True, which='both', alpha=0.3)
    
    # P2 plot
    ax2.plot(N_mu_values, P2_values, 's-', linewidth=2.5, markersize=8, color='#ff7f0e')
    ax2.set_xlabel('Number of μ integration points', fontsize=12)
    ax2.set_ylabel('P2 value', fontsize=12)
    ax2.set_title(f'{model}: P2 Convergence', fontsize=13)
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Test convergence:
test_simpson_convergence(planck_params, model='LCDM')
