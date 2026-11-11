"""
LSS hooks into stock Kilosort4 (installed from PyPI, no fork).

The training phase needs a few things Kilosort does not produce on its own:
    - grouped common median reference (car_groups, see cmr_groups.py)
    - Wall3.npy, wPCA.npy, ctc.npy: the templates and projections that
      template_matching.extract() matches against, which the CUDA sorter
      reproduces at runtime
    - cluster_centroids.npy: mean (x, y) position of each final cluster

These reach into Kilosort internals (BinaryFiltered.filter, detect_spikes), so
this module is tied to KILOSORT_VERSION. Bumping Kilosort means re-checking
this file against the new release and re-validating the training outputs.
"""
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import torch

KILOSORT_VERSION = "4.0.13"


def check_kilosort_version():
    installed = version("kilosort")
    if installed != KILOSORT_VERSION:
        raise RuntimeError(
            f"LSS training is validated against kilosort=={KILOSORT_VERSION}, "
            f"but kilosort=={installed} is installed. "
            f"Install it with: pip install kilosort=={KILOSORT_VERSION}")


@contextmanager
def grouped_car(car_groups):
    """
    While active, every kilosort.io.BinaryFiltered that Kilosort creates
    subtracts the median per channel group instead of across all channels.
    The file objects keep that behavior after the context exits.
    """
    from kilosort import io as ks_io
    from kilosort.preprocessing import get_drift_matrix, fft_highpass
    from torch.fft import fft, ifft, fftshift

    groups = np.asarray(car_groups).ravel()
    if np.unique(groups).size <= 1:
        yield
        return
    group_index = [torch.from_numpy(np.flatnonzero(groups == g)) for g in np.unique(groups)]

    class GroupedCARBinaryFiltered(ks_io.BinaryFiltered):
        # Copy of BinaryFiltered.filter from kilosort 4.0.13; only the CAR step differs.
        def filter(self, X, ops=None, ibatch=None):
            if self.chan_map is not None:
                X = X[self.chan_map]

            if self.invert_sign:
                X = X * -1

            X = X - X.mean(1).unsqueeze(1)
            if self.do_CAR:
                # median across the channels of each group
                for idx in group_index:
                    idx = idx.to(X.device)
                    X[idx] = X[idx] - torch.median(X[idx], 0)[0]

            if self.hp_filter is not None:
                fwav = fft_highpass(self.hp_filter, NT=X.shape[1])
                X = torch.real(ifft(fft(X) * torch.conj(fwav)))
                X = fftshift(X, dim=-1)

            if self.artifact_threshold < np.inf:
                if torch.any(torch.abs(X) >= self.artifact_threshold):
                    return torch.zeros_like(X)

            if self.whiten_mat is not None:
                if self.dshift is not None and ops is not None and ibatch is not None:
                    M = get_drift_matrix(ops, self.dshift[ibatch], device=self.device)
                    X = (M @ self.whiten_mat) @ X
                else:
                    X = self.whiten_mat @ X
            return X

    original = ks_io.BinaryFiltered
    ks_io.BinaryFiltered = GroupedCARBinaryFiltered
    try:
        yield
    finally:
        ks_io.BinaryFiltered = original


def detect_spikes(ops, results_dir, device, bfile, tic0=np.nan, progress_bar=None):
    """
    kilosort.run_kilosort.detect_spikes, plus saving the matching templates
    (Wall3), PCA basis (wPCA) and template cross-terms (ctc) to results_dir.
    """
    from kilosort import spikedetect, clustering_qr, template_matching
    from kilosort.run_kilosort import logger

    results_dir = Path(results_dir)

    tic = time.time()
    logger.info(' ')
    logger.info('Extracting spikes using templates')
    logger.info('-'*40)
    st0, tF, ops = spikedetect.run(ops, bfile, device=device, progress_bar=progress_bar)
    tF = torch.from_numpy(tF)
    logger.info(f'{len(st0)} spikes extracted in {time.time()-tic : .2f}s; ' +
                f'total {time.time()-tic0 : .2f}s')
    if len(st0) == 0:
        raise ValueError('No spikes detected, cannot continue sorting.')

    tic = time.time()
    logger.info(' ')
    logger.info('First clustering')
    logger.info('-'*40)
    clu, Wall = clustering_qr.run(ops, st0, tF, mode='spikes', device=device,
                                  progress_bar=progress_bar)
    Wall3 = template_matching.postprocess_templates(Wall, ops, clu, st0, device=device)
    logger.info(f'{clu.max()+1} clusters found, in {time.time()-tic : .2f}s; ' +
                f'total {time.time()-tic0 : .2f}s')

    np.save(results_dir / 'Wall3.npy', Wall3.cpu().numpy())
    np.save(results_dir / 'wPCA.npy', ops['wPCA'].cpu().numpy())
    ctc = template_matching.prepare_matching(ops, Wall3)
    np.save(results_dir / 'ctc.npy', ctc.cpu().numpy())

    tic = time.time()
    logger.info(' ')
    logger.info('Extracting spikes using cluster waveforms')
    logger.info('-'*40)
    st, tF, ops = template_matching.extract(ops, bfile, Wall3, device=device,
                                            progress_bar=progress_bar)
    logger.info(f'{len(st)} spikes extracted in {time.time()-tic : .2f}s; ' +
                f'total {time.time()-tic0 : .2f}s')

    return st, tF, Wall, clu


def save_cluster_centroids(results_dir, st, clu, tF, ops):
    """Mean (x, y) spike position per cluster -> cluster_centroids.npy ({cluster: (x, y)})."""
    from kilosort.postprocessing import compute_spike_positions

    xs, ys = compute_spike_positions(st, tF, ops)
    positions = np.vstack([xs, ys]).T
    centroids = {}
    for c in np.unique(clu):
        centroid = positions[clu == c].mean(axis=0)
        centroids[c] = (centroid[0], centroid[1])
    np.save(Path(results_dir) / 'cluster_centroids.npy', centroids)
    return centroids
