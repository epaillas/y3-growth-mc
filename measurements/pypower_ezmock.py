import fitsio
from astropy.io import fits
from pathlib import Path
import argparse
import numpy as np
from pypower import mpi, CatalogFFTPower
from mpytools import Catalog
from acm import setup_logging
from cosmoprimo.fiducial import AbacusSummit
import glob
import time
import sys


def data_fn(tracer='LRG', phase_idx=0, redshift=0.8):
    """
    Return the filename of the data for a given tracer, phase index and redshift.
    """
    base_dir = '/global/cfs/cdirs/desicollab/cosmosim/SecondGenMocks/EZmock/CubicBox_6Gpc/'
    data_dir = Path(base_dir) / f'{tracer}/z{redshift:.3f}/{phase_idx:04}/'
    data_fn = data_dir / f'EZmock_{tracer}_z{redshift:.3f}_AbacusSummit_base_c000_ph000_{phase_idx:04}.*.fits.gz'
    data_fns = glob.glob(str(data_fn))
    return data_fns

def read_data(tracer='LRG', phase_idx=0, redshift=0.8, los='z'):
    """
    Read the (redshift-space) data positions for a given tracer, phase index and redshift.
    """
    fns = data_fn(tracer, phase_idx, redshift)
    pos_all = []
    for i, fn in enumerate(fns):
        # distribute files into ranks
        if i % mpicomm.size != mpicomm.rank:
            continue
        data = fitsio.read(fn)
        hubble = 100 * cosmo.efunc(redshift)
        scale_factor = 1 / (1 + redshift)
        if los == 'x':
            rsd = data['vx'] / (hubble * scale_factor)
            pos = np.c_[data['x'] + rsd, data['y'], data['z']] % boxsize
        elif los == 'y':
            rsd = data['vy'] / (hubble * scale_factor)
            pos = np.c_[data['x'], data['y'] + rsd, data['z']] % boxsize
        elif los == 'z':
            rsd = data['vz'] / (hubble * scale_factor)
            pos = np.c_[data['x'], data['y'], data['z'] + rsd] % boxsize
        pos_all.append(pos)
    # concatenate if not empty
    if len(pos_all) > 0:
        pos_all = np.concatenate(pos_all)
    # # now concatenate all positions across ranks
    # pos_all = mpi.mpi_allgather(pos_all, comm=mpicomm)
    return pos_all


setup_logging()

mpicomm = mpi.COMM_WORLD
mpiroot = 0

# parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--tracer", type=str, default='LRG')
parser.add_argument("--redshift", type=float, default=0.8)
parser.add_argument("--start_phase", type=int, default=1)
parser.add_argument("--n_phase", type=int, default=1)
args = parser.parse_args()
tracer = args.tracer
redshift = args.redshift
phases = list(range(args.start_phase, args.start_phase + args.n_phase))

# define cosmology
cosmo = AbacusSummit()

los = 'z'
boxsize = 6000
nmesh = 768

# measure bispectrum from data
for phase_idx in phases:
    t0 = time.time()
    data_positions = read_data(tracer=tracer, phase_idx=phase_idx, redshift=redshift, los=los)
    if mpicomm.rank == mpiroot:
        print(f'Read {len(data_positions)} phase {phase_idx} in {time.time() - t0:.2f} seconds')
    # else:
        # data_positions = None

    power = CatalogFFTPower(data_positions1=data_positions, edges={'step': 0.001}, los=los, nmesh=nmesh,
                            boxsize=boxsize, resampler='tsc', interlacing=3, position_type='pos',
                            wrap=True, mpicomm=mpicomm, mpiroot=None).poles

    # save results
    base_dir = '/pscratch/sd/e/epaillas/desi/gqc-y3-growth/mock_challenge/pypower/spectrum/SecondGenMocks/CubicBox/'
    save_dir = Path(base_dir) / f'{tracer}/z{redshift:.3f}/EZmock_6Gpc/ph{phase_idx:04}/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / f'spectrum_ph{phase_idx:04}.npy'
    if mpicomm.rank == mpiroot:
        power.save(save_fn)