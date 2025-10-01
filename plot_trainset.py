from jaxpower import read
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_spectrum_fns(cosmo_idx=0, phase_idx=0):
    base_dir = Path(f'/pscratch/sd/e/epaillas/desi/gqc-y3-growth/mock_challenge/jaxpower/spectrum/training_set/yuan23/z0.8/ngal_8.5e-4/ap/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/')
    handle = f'mesh2_spectrum_poles_c{cosmo_idx:03}_hod???.h5'
    fns = sorted(base_dir.glob(handle))
    return fns

def get_bispectrum_fns(cosmo_idx=0, phase_idx=0):
    base_dir = Path(f'/pscratch/sd/e/epaillas/desi/gqc-y3-growth/mock_challenge/jaxpower/bispectrum/training_set/yuan23/z0.8/ngal_8.5e-4/ap/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/')
    handle = f'mesh3_spectrum_poles_scoccimarro_c{cosmo_idx:03}_hod???.h5'
    fns = sorted(base_dir.glob(handle))
    return fns

def read_spectrum(filename):
    data = read(filename)
    kmin, kmax = 0.01, 0.7
    data = data.select(k=slice(0, None, 5)).select(k=(kmin, kmax))
    poles = [data.get(ell) for ell in (0, 2, 4)]
    k = poles[0].coords('k')
    return k, poles

def read_bispectrum(filename):
    data = read(filename)
    # xlim = (0.03, 0.3)
    # data = data.select(k=slice(0, None, 2)).select(k=xlim)
    data = data.select(k=slice(0, None, 1))
    poles = [data.get(ell) for ell in (0, 2)]
    x = np.prod(poles[0].coords('k', center='mid'), axis=-1)
    print(poles[0].coords('k'))
    return x, poles

def plot_spectrum():
    pk_fns = np.concatenate([get_spectrum_fns(cosmo_idx=i) for i in range(5)])
    k, _ = read_spectrum(pk_fns[0])
    poles = [read_spectrum(fn)[1] for fn in pk_fns]

    fig, ax = plt.subplots(figsize=(4, 3))
    for pole in poles:
        for ell in (0,):
            ax.plot(k, k*pole[ell//2], ls='-', lw=1.0)
    ax.set_xlabel(r'$k$ [h/Mpc]')
    ax.set_ylabel(r'$k P_\ell(k)$ [h$^{-2}$ Mpc$^{2}$]')
    plt.tight_layout()
    plt.savefig('spectrum_training.png', dpi=300, bbox_inches='tight')

def plot_bispectrum():
    bk_fns = np.concatenate([get_bispectrum_fns(cosmo_idx=i) for i in range(5)])
    x, _ = read_bispectrum(bk_fns[0])
    poles = [read_bispectrum(fn)[1] for fn in bk_fns]

    fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    for pole in poles:
        for ell in (0, 2):
            ax[ell//2].plot(np.arange(len(x)), x * pole[ell//2], ls='-', lw=1.0)
    ax[1].set_xlabel(r'$k_1 k_2 k_3$ [h$^3$/Mpc$^3$]', fontsize=13)
    fig.supylabel(r'$(k_1 k_2 k_3) B_\ell(k_1, k_2, k_3)$ [h$^{-3}$ Mpc$^{3}$]', fontsize=13)
    plt.tight_layout()
    plt.savefig('bispectrum_training.png', dpi=300, bbox_inches='tight')



if __name__ == '__main__':

    plot_spectrum()
    plot_bispectrum()


