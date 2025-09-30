from mockfactory import setup_logging
from pathlib import Path
import pandas
import numpy as np
import glob
import fitsio
import os
import sys
import time
import argparse


def get_hod_fns(cosmo=0, phase=0, redshift=0.8):
    """
    Get the list of HOD file names for a given cosmology,
    phase, and redshift.
    """
    base_dir = '/pscratch/sd/e/epaillas/desi/gqc-y3-growth/mock_challenge/hods/yuan23/'
    hod_dir = Path(base_dir) / f'z{redshift:.1f}/ngal_8.5e-4/ap/c{cosmo:03}_ph{phase:03}/seed{seed_idx}/'
    hod_fns = glob.glob(str(Path(hod_dir) / f'hod*.fits'))
    return hod_fns

def get_hod_positions(filename, los='z'):
    """Get redshift-space positions from a HOD file."""
    hod, header = fitsio.read(filename, header=True)
    qpar, qperp = header['Q_PAR'], header['Q_PERP']
    if los == 'x':
        pos = np.c_[hod['X_RSD'], hod['Y_PERP'], hod['Z_PERP']]
        boxsize = np.array([2000/qpar, 2000/qperp, 2000/qperp])
        return pos, boxsize
    elif los == 'y':
        pos = np.c_[hod['X_PERP'], hod['Y_RSD'], hod['Z_PERP']]
        boxsize = np.array([2000/qperp, 2000/qpar, 2000/qperp])
        return pos, boxsize
    elif los == 'z':
        pos = np.c_[hod['X_PERP'], hod['Y_PERP'], hod['Z_RSD']]
        boxsize = np.array([2000/qperp, 2000/qperp, 2000/qpar])
        return pos, boxsize

def compute_spectrum(output_fn, positions, ells=(0, 2, 4), los='z', **attrs):
    from jaxpower import (MeshAttrs, ParticleField, FKPField, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum, compute_fkp2_shotnoise)
    print(f'Processing cosmo {cosmo_idx}, phase {phase_idx}, seed {seed_idx}, hod {hod_idx}')
    t0 = time.time()
    mattrs = MeshAttrs(**attrs)
    data = ParticleField(positions, attrs=mattrs, exchange=True, backend='jax')
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mean = mesh.mean()
    mesh = mesh - mean
    bin = BinMesh2SpectrumPoles(mesh.attrs, edges={'step': 0.001}, ells=ells)
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los)
    num_shotnoise = compute_fkp2_shotnoise(data, bin=bin)
    spectrum = spectrum.clone(norm=[pole.values('norm') * mean**2 for pole in spectrum], num_shotnoise=num_shotnoise)
    # spectrum.attrs.update(mesh=dict(mesh.attrs), los=los)
    # print(spectrum.attrs)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Power spectrum done in {t1 - t0:.2f} s.')
        print(f'Saving to {output_fn}')
    spectrum.write(output_fn)

def compute_box_bispectrum(output_fn, positions, basis='scoccimarro', los='z', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum, MeshAttrs)
    t0 = time.time()
    mattrs = MeshAttrs(**attrs)
    data = ParticleField(positions, attrs=mattrs, exchange=True, backend='jax')
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mean = mesh.mean()
    mesh = mesh - mean
    ells = [(0, 0, 0), (0, 0, 2)] if 'sugiyama' in basis else [0, 2]
    # bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01, 'max': 0.201}, basis=basis, ells=ells, buffer_size=2)
    bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.02, 'max': 0.32}, basis=basis, ells=ells, buffer_size=2)
    #jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'], donate_argnums=[0])
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(data, los=los, bin=bin, **kw)
    mesh = data.paint(**kw, out='real')
    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    spectrum = spectrum.clone(norm=[pole.values('norm') * mean**3 for pole in spectrum], num_shotnoise=num_shotnoise)
    # spectrum.attrs.update(mesh=dict(mesh.attrs), los=los)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Bispectrum done in {t1 - t0:.2f} s.')
        print(f'Saving to {output_fn}')
    spectrum.write(output_fn)

if __name__ == '__main__':

    is_distributed = True
    if is_distributed:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        jax.distributed.initialize()
    from jax import config
    config.update('jax_enable_x64', True)
    from jaxpower.mesh import create_sharding_mesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--n_seed", type=int, default=1)

    args = parser.parse_args()

    phases = list(range(args.start_phase, args.start_phase + args.n_phase))
    cosmos = list(range(args.start_cosmo, args.start_cosmo + args.n_cosmo))
    seeds = list(range(args.start_seed, args.start_seed + args.n_seed))

    setup_logging()

    start_hod = 0
    n_hod = 100
    redshift = 0.8

    todo_stats = ['spectrum', 'bispectrum']

    for cosmo_idx in cosmos:
        for phase_idx in phases:
            for seed_idx in seeds:
                hod_fns = get_hod_fns(cosmo=cosmo_idx, phase=phase_idx, redshift=redshift)

                for hod_fn in hod_fns[start_hod:start_hod+n_hod]:
                    hod_idx = hod_fn.split('.fits')[0].split('hod')[-1]
                    hod_positions, boxsize = get_hod_positions(hod_fn, los='z')

                    box_args = dict(boxsize=boxsize, boxcenter=0.0, meshsize=512, los='z', ells=(0, 2, 4))

                    if 'spectrum' in todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/desi/gqc-y3-growth/mock_challenge/jaxpower/spectrum/training_set/'
                        save_dir += f'yuan23/z{redshift:.1f}/ngal_8.5e-4/ap/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(save_dir) / f'mesh2_spectrum_poles_c{cosmo_idx:03}_hod{hod_idx:03}.h5'
                        if not output_fn.exists():
                            with create_sharding_mesh() as sharding_mesh:
                                compute_spectrum(output_fn, hod_positions, **box_args)

                    if 'bispectrum' in todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/desi/gqc-y3-growth/mock_challenge/jaxpower/bispectrum/training_set/'
                        save_dir += f'yuan23/z{redshift:.1f}/ngal_8.5e-4/ap/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(save_dir) / f'mesh3_spectrum_poles_scoccimarro_c{cosmo_idx:03}_hod{hod_idx:03}.h5'
                        args = box_args | dict(basis='scoccimarro', meshsize=256)
                        args.pop('ells')
                        if not output_fn.exists():
                            with create_sharding_mesh() as sharding_mesh:
                                compute_box_bispectrum(output_fn, hod_positions, **args)

                    # Shut down distributed environment
                    jax.distributed.shutdown()



