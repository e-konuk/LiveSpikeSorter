#!/usr/bin/env python3
"""
subset_templates.py  --  Prototype template-subsetting tool for LSS.

PURPOSE
-------
Produce a reduced copy of a Kilosort-derived `oss_input/` directory that
contains only the N most useful *preclustered* templates, so that lower-spec
GPUs can run the real-time sorter faster.

The dominant GPU step (`matchingPursuit`) convolves the live stream against
`Wall3.npy`, whose first dimension is the *preclustered* template count
(`unclu_T` in OnlineSpikesV2). Its cost scales ~linearly with `unclu_T`, so
trimming `unclu_T` is the lever that actually buys real-time headroom.

This tool subsets ONLY the preclustered (`unclu_T`) space. It leaves the
final-cluster (`T`) space (templates.npy, templateMap.npy,
cluster_centroids_pca.npy) and all channel/PC-space tensors untouched. That
keeps the engine's load-time invariants intact and avoids the (geometric,
runtime-only) preclustered->cluster remapping problem entirely. Some final
clusters may simply stop receiving spikes; that is the intended graceful
degradation.

You do NOT need to re-run Kilosort to use this. It consumes an existing,
already-cropped `oss_input/` and writes a new sibling directory.

!!! IMPORTANT ENGINE-COUPLING WARNING !!!
------------------------------------------
Output spike `template_id`s are assigned by OnlineSpikesV2::closestCluster(),
which finds the nearest of the FINAL-cluster centroids (cluster_centroids.npy,
T=629). That assignment is independent of which preclustered templates are
active, so subsetting needs NO downstream id remapping.

HOWEVER, closestCluster()'s scan loop is bounded by `unclu_T`, not by the
centroid count. Reducing unclu_T below T therefore makes closestCluster scan
only the first `unclu_T` of the T centroids -> spikes belonging to clusters
[unclu_T, T) get misassigned. So this offline tool is only fully correct down
to N = T (768 -> 629 here, ~18% conv reduction). To subset BELOW T you must
first apply the one-line engine fix: change closestCluster()'s loop bound from
`unclu_T` to `(long)xs.size()` (== T). That fix also removes a pre-existing
out-of-bounds read/write (today unclu_T=768 > xs.size()=629). Until that ships,
this script refuses N < T unless --allow-below-T is passed.

USAGE
-----
    # Inspect first (no writes): print every file's shape + planned action.
    python subset_templates.py --input  C:\\...\\oss_input \\
                               --n_templates 300 --dry-run

    # Produce a reduced directory.
    python subset_templates.py --input  C:\\...\\oss_input \\
                               --output C:\\...\\oss_input_n300 \\
                               --n_templates 300 --metric spikes

Then point LSS at the --output directory (KS Output Dir / --oss_input).
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# File classification
#
#   unclu_T  = Wall3.npy.shape[0]      (preclustered template count)
#   T        = templates.npy.shape[0]  (final cluster count)
#   K        = Wall3.npy.shape[1]      (principal components)
#   M        = templates.npy.shape[1]  (samples per template)
#   C        = templates.npy.shape[2]  (channels)
#
# ---------------------------------------------------------------------------

UNCLU_T_AXES = {
    "Wall3.npy": (0,),                            # [unclu_T, K, C]
    "ctc.npy": (0, 1),                            # [unclu_T, unclu_T, 2M+1]
    "ctc_permuted.npy": (0, 1),                   # [unclu_T, unclu_T, 2M+1]; not engine-loaded, kept consistent
    "iU.npy": (0,),                               # [unclu_T]
    "Ucc.npy": (1,),                              # [numNearestChans, unclu_T, K] -- see d_Ucc indexing OnlineSpikesV2.cu:162-164
    "preclustered_template_waveforms.npy": (0,),  # [unclu_T, M, C]
}


# Array used to RANK templates by activity
RANK_SOURCE = "spike_detection_templates.npy"

# Files to copy verbatim, listed for documentation
COPY_VERBATIM_HINT = (
    "templates.npy", "templateMap.npy", "cluster_centroids_pca.npy",
    "whiteningMat.npy", "channelMap.npy", "drift_matrix.npy", "iCC.npy",
    "wPCA.npy", "wPCA_permuted.npy", "hp_filter.npy", "xc.npy", "yc.npy",
    "misc.txt", "ops.npz",
)


def _load_shapes(in_dir: Path):
    """Reads the dimensions from templates.npy and Wall3.npy."""
    templates = np.load(in_dir / "templates.npy")
    wall3 = np.load(in_dir / "Wall3.npy")

    if templates.ndim != 3:
        sys.exit(f"templates.npy expected 3-D [T,M,C], got shape {templates.shape}")
    if wall3.ndim != 3:
        sys.exit(f"Wall3.npy expected 3-D [unclu_T,K,C], got shape {wall3.shape}")

    T, M, C = templates.shape
    unclu_T, K, C_w = wall3.shape
    if C_w != C:
        sys.exit(f"Channel mismatch: templates C={C} but Wall3 C={C_w}")

    return dict(T=int(T), M=int(M), C=int(C), unclu_T=int(unclu_T), K=int(K))


def _choose_keep(in_dir: Path, dims: dict, n: int, metric: str):
    """Return a sorted list of preclustered template indices to keep."""
    unclu_T = dims["unclu_T"]
    if n >= unclu_T:
        return list(range(unclu_T)), "all"

    if metric == "spikes":
        sp_path = in_dir / RANK_SOURCE
        if sp_path.exists():
            sp = np.load(sp_path).astype(np.int64).ravel()
            if sp.size and int(sp.max()) < dims["T"]:
                sys.exit(
                    f"[abort] {RANK_SOURCE} max index {int(sp.max())} < T ({dims['T']}); "
                    f"it looks like FINAL-cluster space, not preclustered. Ranking would "
                    f"silently drop templates [T, unclu_T). Check the file."
                )
            sp = sp[(sp >= 0) & (sp < unclu_T)]
            counts = np.bincount(sp, minlength=unclu_T)
            order = np.argsort(counts)[::-1]  # find most active first
            ranked = "spike count"
        else:
            print("  [warn] spike_templates.npy not found; falling back to norm")
            metric = "norm"

    if metric == "norm":
        wall3 = np.load(in_dir / "Wall3.npy").reshape(unclu_T, -1)
        energy = np.linalg.norm(wall3, axis=1)
        order = np.argsort(energy)[::-1]  # highest energy first
        ranked = "template L2 norm"

    keep = sorted(int(i) for i in order[:n])
    print(f"  selected {len(keep)} / {unclu_T} preclustered templates by {ranked}")
    return keep, ranked


def _plan(in_dir: Path, dims: dict):
    """Return list of (filename, action, detail) for dry-run reporting"""
    rows = []
    for name in sorted(os.listdir(in_dir)):
        p = in_dir / name
        if p.is_dir():
            rows.append((name, "skip (dir)", ""))
            continue
        if name in UNCLU_T_AXES:
            arr = np.load(p, allow_pickle=False)
            rows.append((name, "subset", f"shape {arr.shape}, axes {UNCLU_T_AXES[name]}"))
        elif name.endswith(".npy"):
            arr = np.load(p, allow_pickle=False)
            rows.append((name, "copy verbatim", f"shape {arr.shape}"))
        else:
            rows.append((name, "copy verbatim", ""))
    return rows


def _subset_axes(arr: np.ndarray, axes, keep_idx: np.ndarray, unclu_T: int, name: str):
    for ax in axes:
        real_ax = ax if ax >= 0 else arr.ndim + ax
        if arr.shape[real_ax] != unclu_T:
            sys.exit(
                f"[abort] {name}: axis {ax} length {arr.shape[real_ax]} "
                f"!= unclu_T {unclu_T}. Encoded rule is wrong for this data; "
                f"re-run with --dry-run and verify axis order before trusting."
            )
        arr = np.take(arr, keep_idx, axis=real_ax)
    return arr


def subset_oss_input_inplace(oss_dir, n, metric="spikes", allow_below_t=False):
    """Subset the preclustered (unclu_T) tensors of an oss_input dir IN PLACE.

    Called by run_online_spikes.py right after it regenerates `oss_input`.
    """
    oss_dir = Path(oss_dir)
    dims = _load_shapes(oss_dir)
    unclu_T, T = dims["unclu_T"], dims["T"]

    if not n or n <= 0 or n >= unclu_T:
        print(f"[subset] keeping all {unclu_T} preclustered templates (N={n}, no-op).")
        return {"new_unclu_T": unclu_T, "applied": False, **dims}

    if n < T:
        if not allow_below_t:
            raise RuntimeError(
                f"[subset] max_templates {n} < final-cluster count T ({T}). "
                f"closestCluster() scans only the first unclu_T centroids, so N < T "
                f"misassigns spikes until the one-line engine fix ships (loop bound "
                f"unclu_T -> xs.size()). Pass allow_below_t=True only on a rebuilt engine."
            )
        print(f"[subset] WARNING: N={n} is below the final-cluster count T={T}. "
              f"This is only correct on an OnlineSpikes.exe built WITH the "
              f"closestCluster() loop-bound fix (unclu_T -> xs.size()). On an "
              f"unpatched build, spike->cluster assignment will be wrong.")

    keep, ranked = _choose_keep(oss_dir, dims, n, metric)
    keep_idx = np.asarray(keep, dtype=np.int64)

    for name, axes in UNCLU_T_AXES.items():
        p = oss_dir / name
        if not p.exists():
            continue
        arr = np.load(p, allow_pickle=False)
        arr = _subset_axes(arr, axes, keep_idx, unclu_T, name)
        np.save(p, arr)

    (oss_dir / "subset_manifest.json").write_text(json.dumps({
        "original_unclu_T": unclu_T, "new_unclu_T": len(keep),
        "T_unchanged": T, "metric": ranked, "kept_indices": keep,
    }, indent=2))

    _verify(oss_dir, len(keep), dims)
    print(f"[subset] reduced preclustered templates {unclu_T} -> {len(keep)} (by {ranked}).")
    return {"new_unclu_T": len(keep), "applied": True, "kept_indices": keep, **dims}


def run(in_dir: Path, out_dir: Path, n: int, metric: str, dry_run: bool, allow_below_t: bool):
    dims = _load_shapes(in_dir)
    print(f"Detected: unclu_T={dims['unclu_T']} (preclustered), T={dims['T']} "
          f"(final clusters), K={dims['K']}, M={dims['M']}, C={dims['C']}")

    if dry_run:
        print("\n-- DRY RUN: planned actions --")
        for name, action, detail in _plan(in_dir, dims):
            print(f"  {name:42s} {action:26s} {detail}")
        return

    if n >= dims["unclu_T"]:
        sys.exit(f"--n_templates {n} >= unclu_T {dims['unclu_T']}: nothing to do.")
    if n <= 0:
        sys.exit("--n_templates must be a positive integer.")
    if n < dims["T"] and not allow_below_t:
        sys.exit(
            f"--n_templates {n} < T ({dims['T']}). closestCluster() scans only the\n"
            f"first unclu_T centroids, so N < T misassigns spikes until the one-line\n"
            f"engine fix ships (see module docstring). Re-run with --allow-below-T to\n"
            f"override (e.g. for a build that already has the fix)."
        )

    keep, ranked = _choose_keep(in_dir, dims, n, metric)
    keep_idx = np.asarray(keep, dtype=np.int64)
    unclu_T = dims["unclu_T"]

    if out_dir.exists():
        sys.exit(f"--output {out_dir} already exists; refusing to overwrite.")
    out_dir.mkdir(parents=True)

    actions = {}
    for name in sorted(os.listdir(in_dir)):
        src = in_dir / name
        if src.is_dir():
            continue
        dst = out_dir / name

        if name in UNCLU_T_AXES:
            arr = np.load(src, allow_pickle=False)
            arr = _subset_axes(arr, UNCLU_T_AXES[name], keep_idx, unclu_T, name)
            np.save(dst, arr)
            actions[name] = f"subset axes {UNCLU_T_AXES[name]} -> {arr.shape}"
        else:
            shutil.copy2(src, dst)
            actions[name] = "copy verbatim"

    manifest = {
        "source": str(in_dir),
        "original_unclu_T": unclu_T,
        "new_unclu_T": len(keep),
        "T_unchanged": dims["T"],
        "metric": ranked,
        "kept_indices": keep,
        "actions": actions,
    }
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))

    _verify(out_dir, len(keep), dims)
    print(f"\nDone. Wrote {len(keep)}-template directory to {out_dir}")
    print("Point LSS (--oss_input / KS Output Dir) at it. No Kilosort re-run needed.")


def _verify(out_dir: Path, new_unclu_T: int, dims: dict):
    """Re-check the engine's load-time invariants on the produced directory."""
    print("\n-- Verifying output invariants --")
    checks = {
        "Wall3.npy": ("shape0", new_unclu_T),
        "ctc.npy": ("shape01", new_unclu_T),
        "ctc_permuted.npy": ("shape01", new_unclu_T),
        "iU.npy": ("len", new_unclu_T),
        "Ucc.npy": ("shape1", new_unclu_T),
        "preclustered_template_waveforms.npy": ("shape0", new_unclu_T),
        "templates.npy": ("shape0", dims["T"]),              # unchanged 
        "templateMap.npy": ("len", dims["T"]),               # unchanged
        "cluster_centroids.npy": ("shape0", dims["T"]),      # unchanged 
        "cluster_centroids_pca.npy": ("shape0", dims["T"]),  # unchanged
    }
    ok = True
    for name, (kind, expected) in checks.items():
        p = out_dir / name
        if not p.exists():
            print(f"  [skip] {name} not present in source")
            continue
        arr = np.load(p, allow_pickle=False)
        if kind == "shape0":
            got = arr.shape[0]
        elif kind == "shape01":
            got = arr.shape[0] if arr.shape[0] == arr.shape[1] else -1
        elif kind == "shape1":
            got = arr.shape[1]
        elif kind == "len":
            got = arr.shape[0]
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  {name:42s} {kind:10s} expected {expected:>6}  got {got:>6}  [{status}]")
    if not ok:
        sys.exit("[abort] verification failed; output directory is NOT safe to use.")


def main():
    ap = argparse.ArgumentParser(description="Subset LSS preclustered templates.")
    ap.add_argument("--input", required=True, type=Path, help="source oss_input dir")
    ap.add_argument("--output", type=Path, help="destination dir (required unless --dry-run)")
    ap.add_argument("--n_templates", type=int, default=0, help="number of preclustered templates to keep")
    ap.add_argument("--metric", choices=("spikes", "norm"), default="spikes",
                    help="selection criterion: 'spikes' (most active) or 'norm' (highest energy)")
    ap.add_argument("--dry-run", action="store_true", help="inspect shapes/actions without writing")
    ap.add_argument("--allow-below-T", action="store_true",
                    help="permit N < final-cluster count T (only safe with the closestCluster engine fix)")
    args = ap.parse_args()

    if not args.input.is_dir():
        sys.exit(f"--input {args.input} is not a directory")
    if not args.dry_run and args.output is None:
        sys.exit("--output is required unless --dry-run")

    run(args.input, args.output, args.n_templates, args.metric, args.dry_run, args.allow_below_T)


if __name__ == "__main__":
    main()
