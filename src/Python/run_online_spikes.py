# Only numpy + stdlib are needed to open this GUI and launch OnlineSpikes.exe
# against an existing oss_input/ (requirements-runtime.txt). torch, kilosort,
# qtpy, and matplotlib are only needed to (re)run Kilosort4 -- see
# requirements-training.txt -- and are imported lazily, inside the functions
# that actually use them, so launching doesn't require installing them.
import sys
import pathlib
from pathlib import Path
from crop_methods import crop_kilosort_output, parse_bin_meta_file
from subset_templates import subset_oss_input_inplace, parse_channel_range
import numpy as np
import subprocess
import time
import os
import shlex
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import json

# -------------------------------
# Persistence config
# -------------------------------
STATE_FILE = Path(__file__).parent / "multi_gui_state.json"

# -------------------------------
# GUI for Multiple Sorters
# -------------------------------
root = tk.Tk()
root.title("Live Spike Sorter \u2014 Configuration")

# Number of sorters
num_sorters_var = tk.IntVar(value=1)
spin = tk.Spinbox(root, from_=1, to=16, textvariable=num_sorters_var, width=5, command=lambda: update_tabs())
tk.Label(root, text="Number of sorters:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# Global Data Acquisition parameters (shared across all sorters: one SpikeGLX
# stream connection for the whole process)
sglx_host_var = tk.StringVar(value="127.0.0.1")
sglx_port_var = tk.StringVar(value="4142")

daq_frame = tk.Frame(root, borderwidth=1, relief="groove")
daq_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="w")

tk.Label(daq_frame, text="Data Acquisition (shared)", font=("TkDefaultFont", 9, "bold")).grid(
    row=0, column=0, columnspan=4, padx=5, pady=(5, 2), sticky="w"
)

tk.Label(daq_frame, text="SGLX Host:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
tk.Entry(daq_frame, textvariable=sglx_host_var, width=15).grid(row=1, column=1, sticky="w")
tk.Button(daq_frame, text="?", command=lambda: show_hint("SGLX_HOST"), width=3).grid(row=1, column=3)

tk.Label(daq_frame, text="SGLX Port:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
tk.Entry(daq_frame, textvariable=sglx_port_var, width=15).grid(row=2, column=1, sticky="w")
tk.Button(daq_frame, text="?", command=lambda: show_hint("SGLX_PORT"), width=3).grid(row=2, column=3)

# Drift-correction parameters (shared across all sorters)
drift_enabled_var = tk.BooleanVar(value=False)
drift_window_var = tk.StringVar(value="10")
drift_max_shift_var = tk.StringVar(value="50")
drift_retrain_threshold_var = tk.StringVar(value="50")

drift_frame = tk.Frame(root, borderwidth=1, relief="groove")
drift_frame.grid(row=1, column=4, columnspan=4, padx=5, pady=5, sticky="w")

tk.Label(drift_frame, text="Drift Correction (shared)", font=("TkDefaultFont", 9, "bold")).grid(
    row=0, column=0, columnspan=4, padx=5, pady=(5, 2), sticky="w"
)

tk.Checkbutton(drift_frame, text="Real-time drift estimation", variable=drift_enabled_var).grid(
    row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w"
)
tk.Button(drift_frame, text="?", command=lambda: show_hint("DRIFT_ENABLED"), width=3).grid(row=1, column=3)

tk.Label(drift_frame, text="Window (s):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
tk.Entry(drift_frame, textvariable=drift_window_var, width=10).grid(row=2, column=1, sticky="w")
tk.Button(drift_frame, text="?", command=lambda: show_hint("DRIFT_WINDOW"), width=3).grid(row=2, column=3)

tk.Label(drift_frame, text="Max shift (um):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
tk.Entry(drift_frame, textvariable=drift_max_shift_var, width=10).grid(row=3, column=1, sticky="w")
tk.Button(drift_frame, text="?", command=lambda: show_hint("DRIFT_MAX_SHIFT"), width=3).grid(row=3, column=3)

tk.Label(drift_frame, text="Retrain threshold (um):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
tk.Entry(drift_frame, textvariable=drift_retrain_threshold_var, width=10).grid(row=4, column=1, sticky="w")
tk.Button(drift_frame, text="?", command=lambda: show_hint("DRIFT_RETRAIN_THRESHOLD"), width=3).grid(row=4, column=3)

# Notebook for sorter tabs
toolkit = ttk.Notebook(root)
toolkit.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

# Per-sorter storage
base_path_vars, ks_output_dir_vars = [], []
bin_file_vars, meta_file_vars, chanmap_file_vars = [], [], []
rerun_ks_vars, sdm_vars = [], []
sdm_ip_vars, sdm_port_vars = [], []
sdm_subset_vars, sdm_trigger_z_vars, sdm_baseline_min_seconds_vars, sdm_trigger_bin_ms_vars = [], [], [], []
sdm_processor_vars = []
sdm_mode_vars, sdm_offset_vars, sdm_boundary_us_vars = [], [], []
# Per-population (FS/RS)
sdm_offset_fs_low_vars, sdm_offset_fs_high_vars = [], []
sdm_offset_rs_low_vars, sdm_offset_rs_high_vars = [], []
sdm_z_fs_low_vars, sdm_z_fs_high_vars = [], []
sdm_z_rs_low_vars, sdm_z_rs_high_vars = [], []
cl_offset_entries, cl_z_entries = {}, {}
max_templates_vars = []
channel_range_vars = []
file_frames, sdm_frames = [], []

# Hints dictionary
HINTS = {
    "BASE": "This directory should contain 'imec_raw' folder for your recording.",
    "KS_OUTPUT": "Select where the Kilosort output is or will be stored.",
    "BIN": "The location of your recording (.bin file).",
    "META": "The location of your recording's metadata (.meta file).",
    "CHANMAP": "The location of your probe's channel map (.mat file).",
    "SDM": "Send decoder output to stimulus display machine?",
    "SDM_PROCESSOR": ("How spike activity is turned into an SDM packet, and the "
                      "transport used. 'zscore' / 'logreg' send a 13-byte UDP packet "
                      "(int8 direction, float32 z, uint64 sampleCt) Use 'zscore' for the "
                      "firing-threshold trigger loop. 'closedloop' splits spikes "
                      "into FS/RS populations and sends a 16-byte packet "
                      "(int32 FS, int32 RS, uint64 sampleCt) for the FS/RS "
                      "early-release loop."),
    "SDM_CLOSEDLOOP": ("Closed-loop FS/RS trigger. Pick ONE stat with its radio; only "
                       "that stat's grid is live. Each stat has FOUR independent knobs: "
                       "the FS and RS populations x the low ('-') and high ('+') "
                       "direction, so FS and RS need not match and the deadband need not "
                       "be symmetric. 'median split': a population reads low when its "
                       "per-bin value < training median - (low offset), high when "
                       "> median + (high offset); an offset >= 0 widens that side of the "
                       "neutral zone, 0 = split at the median on that side. "
                       "'z-score': fires low when z < -(neg |z|) and high when "
                       "z > +(pos |z|), z being the per-bin value standardized by the "
                       "training mean/sd. Leave a field blank to fall back to the legacy "
                       "symmetric value. "
                       "'Waveform boundary (us)' is the FS/RS trough->peak "
                       "cutoff (200 matches AnalysisGUI). The baseline is built "
                       "automatically each launch from Kilosort's training sort "
                       "(spike_times/spike_templates in oss_input) -- no re-streaming. "
                       "'Build closed-loop baseline' just rebuilds it now, e.g. after "
                       "changing the boundary or bin size. The 'Subset' field is set "
                       "automatically to the FS+RS neurons -- you do not type templates "
                       "by hand in closed-loop mode."),
    "MAX_TEMPLATES": ("Limit how many preclustered templates the sorter uses "
                      "(0 = use all). Fewer templates = less GPU work in "
                      "matchingPursuit = better real-time performance on low-spec "
                      "GPUs, at some loss of detection. Applied to oss_input every "
                      "launch."),
    "CHANNEL_RANGE": ("Keep only templates with support on a probe channel range, "
                      "e.g. '100-150' (inclusive probe channel numbers; blank = all "
                      "channels). Use this to focus the sorter on one region of "
                      "cortical space. Composes with Max templates: the channel "
                      "range narrows the field first, then Max templates keeps the "
                      "most active survivors. Applied to oss_input every launch."),
    "SGLX_HOST": ("The IP address of the machine running SpikeGLX. If SpikeGLX is "
                  "running on this same machine, leave as 127.0.0.1. Shared by all "
                  "sorters (they use one SpikeGLX connection)."),
    "SGLX_PORT": ("The port SpikeGLX is streaming on. Shared by all sorters."),
    "DRIFT_ENABLED": ("Enable real-time rigid drift estimation + correction. Each "
                      "batch is scanned with Kilosort's universal templates (on "
                      "whitened, NOT yet drift-corrected data), the depth and "
                      "amplitude of those detections are binned into an activity "
                      "fingerprint over a window, and that fingerprint is registered "
                      "against the training reference to give a vertical shift (um). "
                      "The drift-correction matrix is then rebuilt live. A drift "
                      "trace is shown in the output GUI. Shared by all sorters. "
                      "Requires the universal-template exports in oss_input/ -- "
                      "re-run Kilosort4 once if they are missing."),
    "DRIFT_WINDOW": ("Length in seconds of the window used to estimate drift. "
                     "Longer = more spikes = more robust estimate but slower to "
                     "react. Targets slow drift; 10 s is a good starting point and "
                     "matches the diagnostic's default."),
    "DRIFT_MAX_SHIFT": ("Safety clamp (microns) on the estimated shift. This is a "
                        "rail, not a tuning knob: if it engages, the estimate was "
                        "not trustworthy in the first place. The output GUI reports "
                        "how often it fires."),
    "DRIFT_RETRAIN_THRESHOLD": ("Microns of estimated drift above which the output "
                                "GUI suggests retraining templates. This only shows a "
                                "banner -- it never retrains on its own. Press "
                                "'Retrain templates' in the drift plot to actually "
                                "stop sorting and re-run Kilosort4 on the data "
                                "recorded so far, then relaunch. Distinct from Max "
                                "shift, which clamps the estimator itself."),
}

def show_hint(key):
    messagebox.showinfo("Hint", HINTS[key])


def run_baseline_build(oss_in, boundary="200", bin_ms="100"):
    """Compute FS/RS labels + closed-loop baseline stats for one oss_input dir,
    straight from Kilosort's TRAINING sort (spike_times.npy + spike_templates.npy
    already in oss_input) -- no re-streaming through LSS, no relaunch.

    Returns (ok, message). Fast (<1 s); safe to call automatically after Kilosort.
    """
    oss_in = Path(oss_in)
    needed = ["templates.npy", "spike_times.npy", "spike_templates.npy"]
    missing = [f for f in needed if not (oss_in / f).exists()]
    if missing:
        return False, (f"Missing {', '.join(missing)} in\n{oss_in}\n"
                       "Run Kilosort4 (training) first.")
    sdir = Path(__file__).parent.resolve()
    wf = str(sdir / "waveform_classification.py")
    bl = str(sdir / "closedloop_baseline.py")
    labels = str(oss_in / "rs_fs_labels.csv")
    try:
        out1 = subprocess.run([sys.executable, wf, str(oss_in),
                               "--boundary-us", str(boundary)],
                              capture_output=True, text=True, check=True)
        out2 = subprocess.run([sys.executable, bl, str(oss_in),
                               "--labels", labels, "--bin-ms", str(bin_ms)],
                              capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or e.stdout or str(e))[-1500:]
    return True, (out1.stdout + "\n" + out2.stdout).strip()[-1500:]


def fs_rs_ids_from_labels(oss_in):
    """Sorted FS∪RS template indices from oss_input/rs_fs_labels.csv (or [] if
    absent). This IS the SDM subset for the closed-loop trigger -- only the
    classified FS/RS neurons feed the per-bin high/low analysis."""
    import csv as _csv
    path = Path(oss_in) / "rs_fs_labels.csv"
    if not path.exists():
        return []
    ids = []
    with open(path, newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("label", "").strip().upper() in ("FS", "RS"):
                try:
                    ids.append(int(row["template_index"]))
                except (ValueError, KeyError):
                    pass
    return sorted(ids)


def update_cl_mode(idx):
    """Enable only the four threshold fields that the selected closed-loop stat
    uses: the offset grid for 'median', the trigger-Z grid for 'zscore'. Purely
    cosmetic -- makes it clear which knobs are live -- the run still passes both
    sets of values."""
    mode = sdm_mode_vars[idx].get()
    offs = cl_offset_entries.get(idx) or []
    zs = cl_z_entries.get(idx) or []
    try:
        for e in offs:
            e.config(state=("normal" if mode == "median" else "disabled"))
        for e in zs:
            e.config(state=("normal" if mode == "zscore" else "disabled"))
    except tk.TclError:
        pass   # widget destroyed during a tab rebuild


def build_closedloop_baseline(idx):
    """Button handler: rebuild the baseline for sorter idx from its Kilosort
    training sort, auto-fill the SDM subset with the FS+RS neurons, and pop up the
    result. Use it to re-tune the waveform boundary or bin size without re-running
    Kilosort."""
    oss_in = Path(base_path_vars[idx].get().strip()) / "oss_input"
    boundary = sdm_boundary_us_vars[idx].get().strip() or "200"
    bin_ms = sdm_trigger_bin_ms_vars[idx].get().strip() or "100"
    ok, msg = run_baseline_build(oss_in, boundary, bin_ms)
    if ok:
        ids = fs_rs_ids_from_labels(oss_in)
        sdm_subset_vars[idx].set(",".join(str(x) for x in ids))
        msg += f"\n\nSDM subset auto-set to {len(ids)} FS+RS neurons."
    (messagebox.showinfo if ok else messagebox.showerror)(
        "Closed-loop baseline" + ("" if ok else " failed"), msg)

def browse_directory(idx):
    d = filedialog.askdirectory(initialdir=base_path_vars[idx].get() or str(Path.home()), title="Select BASE_PATH")
    if d: base_path_vars[idx].set(d)

def browse_ks_output_dir(idx):
    d = filedialog.askdirectory(initialdir=ks_output_dir_vars[idx].get() or str(Path.home()),
                               title="Select KS Output Directory")
    if d:
        ks_output_dir_vars[idx].set(d)
        # autopopulate BASE_PATH to parent directory of selected Kilosort directory
        base_path_vars[idx].set(str(Path(d).parent))

def browse_bin_file(idx):
    init = os.path.join(base_path_vars[idx].get().strip(), "imec_raw")
    f = filedialog.askopenfilename(initialdir=init, title="Select Recording Binary File", filetypes=[("Binary files","*.bin"), ("All files","*.*")])
    if f: bin_file_vars[idx].set(f)

def browse_meta_file(idx):
    init = os.path.join(base_path_vars[idx].get().strip(), "imec_raw")
    f = filedialog.askopenfilename(initialdir=init, title="Select Metadata File", filetypes=[("Metadata files","*.meta"),("All files","*.*")])
    if f: meta_file_vars[idx].set(f)

def browse_chanmap_file(idx):
    init = os.path.join(base_path_vars[idx].get().strip(), "imec_raw")
    f = filedialog.askopenfilename(initialdir=init, title="Select Channel Map File", filetypes=[("MAT files","*.mat"),("All files","*.*")])
    if f: chanmap_file_vars[idx].set(f)

def toggle_rerun(idx):
    if rerun_ks_vars[idx].get(): file_frames[idx].grid()
    else: file_frames[idx].grid_remove()

def toggle_sdm(idx):
    if sdm_vars[idx].get(): sdm_frames[idx].grid()
    else: sdm_frames[idx].grid_remove()

def update_tabs():
    n = num_sorters_var.get()
    old = len(base_path_vars)

    # Truncate lists if reducing
    if n < old:
        for lst in (base_path_vars, ks_output_dir_vars, bin_file_vars,
                    meta_file_vars, chanmap_file_vars, rerun_ks_vars,
                    sdm_vars, sdm_ip_vars, sdm_port_vars,
                    sdm_subset_vars, sdm_trigger_z_vars, sdm_baseline_min_seconds_vars, sdm_trigger_bin_ms_vars,
                    sdm_processor_vars,
                    sdm_mode_vars, sdm_offset_vars, sdm_boundary_us_vars,
                    sdm_offset_fs_low_vars, sdm_offset_fs_high_vars,
                    sdm_offset_rs_low_vars, sdm_offset_rs_high_vars,
                    sdm_z_fs_low_vars, sdm_z_fs_high_vars,
                    sdm_z_rs_low_vars, sdm_z_rs_high_vars,
                    max_templates_vars, channel_range_vars, file_frames, sdm_frames):
            del lst[n:]

    # Append new entries if increasing
    for i in range(old, n):
        rerun_ks_vars.append(tk.BooleanVar(value=False))
        base_path_vars.append(tk.StringVar(value=str(Path.home())))
        ks_output_dir_vars.append(tk.StringVar(value=""))
        bin_file_vars.append(tk.StringVar(value=""))
        meta_file_vars.append(tk.StringVar(value=""))
        chanmap_file_vars.append(tk.StringVar(value=""))
        sdm_vars.append(tk.BooleanVar(value=False))
        sdm_ip_vars.append(tk.StringVar(value=""))
        sdm_port_vars.append(tk.StringVar(value=""))
        sdm_subset_vars.append(tk.StringVar(value=""))
        sdm_trigger_z_vars.append(tk.StringVar(value="1.0"))
        sdm_baseline_min_seconds_vars.append(tk.StringVar(value="10.0"))
        sdm_trigger_bin_ms_vars.append(tk.StringVar(value="100"))
        sdm_processor_vars.append(tk.StringVar(value="zscore"))
        sdm_mode_vars.append(tk.StringVar(value="median"))
        sdm_offset_vars.append(tk.StringVar(value="0"))
        sdm_offset_fs_low_vars.append(tk.StringVar(value="0"))
        sdm_offset_fs_high_vars.append(tk.StringVar(value="0"))
        sdm_offset_rs_low_vars.append(tk.StringVar(value="0"))
        sdm_offset_rs_high_vars.append(tk.StringVar(value="0"))
        sdm_z_fs_low_vars.append(tk.StringVar(value="1.0"))
        sdm_z_fs_high_vars.append(tk.StringVar(value="1.0"))
        sdm_z_rs_low_vars.append(tk.StringVar(value="1.0"))
        sdm_z_rs_high_vars.append(tk.StringVar(value="1.0"))
        sdm_boundary_us_vars.append(tk.StringVar(value="200"))
        max_templates_vars.append(tk.StringVar(value="0"))
        channel_range_vars.append(tk.StringVar(value=""))
        file_frames.append(None)
        sdm_frames.append(None)

    # Rebuild all tabs
    for tab in toolkit.tabs():
        toolkit.forget(tab)
    for i in range(n):
        frame = ttk.Frame(toolkit)
        toolkit.add(frame, text=f"Sorter {i+1}")
        build_tab(frame, i)

# Construct UI for a single sorter tab
def build_tab(frame, idx):
    row = 0
    # Rerun checkbox
    tk.Checkbutton(
        frame, text="Rerun Kilosort4", variable=rerun_ks_vars[idx],
        command=lambda i=idx: toggle_rerun(i)
    ).grid(row=row, column=0, columnspan=4,
           padx=5, pady=5, sticky="w")
    row += 1

    # Kilosort output directory
    tk.Label(frame, text="Kilosort Output Directory:").grid(
        row=row, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(frame, textvariable=ks_output_dir_vars[idx], width=50).grid(
        row=row, column=1
    )
    tk.Button(
        frame, text="Browse", command=lambda i=idx: browse_ks_output_dir(i)
    ).grid(row=row, column=2)
    tk.Button(
        frame, text="?", command=lambda i=idx: show_hint("KS_OUTPUT"), width=3
    ).grid(row=row, column=3)
    row += 1

    # BASE_PATH
    tk.Label(frame, text="BASE_PATH:").grid(
        row=row, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(frame, textvariable=base_path_vars[idx], width=50).grid(
        row=row, column=1
    )
    tk.Button(
        frame, text="Browse", command=lambda i=idx: browse_directory(i)
    ).grid(row=row, column=2)
    tk.Button(
        frame, text="?", command=lambda i=idx: show_hint("BASE"), width=3
    ).grid(row=row, column=3)
    row += 1

    # Template selection sub-frame (groups the two template-subsetting filters)
    template_frame = tk.Frame(frame, borderwidth=1, relief="sunken")
    template_frame.grid(row=row, column=0, columnspan=4, padx=5, pady=5, sticky="w")
    row += 1

    tk.Label(template_frame, text="Template Selection", font=("TkDefaultFont", 9, "bold")).grid(
        row=0, column=0, columnspan=4, padx=5, pady=(5, 2), sticky="w"
    )

    # Max templates (0 = use all)
    tk.Label(template_frame, text="Max templates (0 = all):").grid(
        row=1, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(template_frame, textvariable=max_templates_vars[idx], width=10).grid(
        row=1, column=1, sticky="w"
    )
    tk.Button(
        template_frame, text="?", command=lambda i=idx: show_hint("MAX_TEMPLATES"), width=3
    ).grid(row=1, column=3)

    # Channel range (blank = all channels)
    tk.Label(template_frame, text="Channel range (e.g. 100-150):").grid(
        row=2, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(template_frame, textvariable=channel_range_vars[idx], width=10).grid(
        row=2, column=1, sticky="w"
    )
    tk.Button(
        template_frame, text="?", command=lambda i=idx: show_hint("CHANNEL_RANGE"), width=3
    ).grid(row=2, column=3)

    # SDM toggle
    tk.Checkbutton(
        frame, text="SDM?", variable=sdm_vars[idx],
        command=lambda i=idx: toggle_sdm(i)
    ).grid(row=row, column=0, columnspan=4,
           sticky="w", padx=5, pady=5)
    tk.Button(
        frame, text="?", command=lambda i=idx: show_hint("SDM"), width=3
    ).grid(row=row, column=3)
    row += 1

    # SDM subframe
    sdm_frame = tk.Frame(frame, borderwidth=1, relief="sunken")
    sdm_frames[idx] = sdm_frame
    tk.Label(sdm_frame, text="SDM IP Address:").grid(
        row=0, column=0, padx=5, pady=5
    )
    tk.Entry(sdm_frame, textvariable=sdm_ip_vars[idx], width=25).grid(
        row=0, column=1
    )
    tk.Label(sdm_frame, text="Port Number:").grid(
        row=1, column=0
    )
    tk.Entry(sdm_frame, textvariable=sdm_port_vars[idx], width=10).grid(
        row=1, column=1
    )

    tk.Label(sdm_frame, text="Subset (comma-separated templates):").grid(
        row=2, column=0, padx=5, pady=5
    )
    tk.Entry(sdm_frame, textvariable=sdm_subset_vars[idx], width=25).grid(
        row=2, column=1
    )

    tk.Label(sdm_frame, text="Trigger Z:").grid(
        row=3, column=0, padx=5, pady=5
    )
    tk.Entry(sdm_frame, textvariable=sdm_trigger_z_vars[idx], width=10).grid(
        row=3, column=1
    )

    tk.Label(sdm_frame, text="Baseline min seconds:").grid(
        row=4, column=0, padx=5, pady=5
    )
    tk.Entry(sdm_frame, textvariable=sdm_baseline_min_seconds_vars[idx], width=10).grid(
        row=4, column=1
    )

    tk.Label(sdm_frame, text="Trigger bin (ms):").grid(
        row=5, column=0, padx=5, pady=5
    )
    tk.Entry(sdm_frame, textvariable=sdm_trigger_bin_ms_vars[idx], width=10).grid(
        row=5, column=1
    )

    tk.Label(sdm_frame, text="Processor (transport):").grid(
        row=6, column=0, padx=5, pady=5
    )
    tk.OptionMenu(sdm_frame, sdm_processor_vars[idx],
                  "zscore", "logreg", "bincounts", "closedloop").grid(
        row=6, column=1, sticky="w"
    )
    tk.Button(
        sdm_frame, text="?", command=lambda i=idx: show_hint("SDM_PROCESSOR"), width=3
    ).grid(row=6, column=2)

    # --- Closed-loop trigger ---
    tk.Label(sdm_frame, text="— Closed-loop trigger (per-population FS/RS, asymmetric) —").grid(
        row=7, column=0, columnspan=3, pady=(8, 0), sticky="w", padx=5
    )
    tk.Button(
        sdm_frame, text="?", command=lambda i=idx: show_hint("SDM_CLOSEDLOOP"), width=3
    ).grid(row=7, column=3)

    # --- median split grid ---
    tk.Radiobutton(sdm_frame, text="median split", variable=sdm_mode_vars[idx],
                   value="median", command=lambda i=idx: update_cl_mode(i)).grid(
        row=8, column=0, sticky="w", padx=5
    )
    tk.Label(sdm_frame, text="low offset (−)").grid(row=8, column=1, sticky="w")
    tk.Label(sdm_frame, text="high offset (+)").grid(row=8, column=2, sticky="w")
    tk.Label(sdm_frame, text="FS:").grid(row=9, column=0, sticky="e", padx=5)
    off_fs_low = tk.Entry(sdm_frame, textvariable=sdm_offset_fs_low_vars[idx], width=8)
    off_fs_low.grid(row=9, column=1, sticky="w")
    off_fs_high = tk.Entry(sdm_frame, textvariable=sdm_offset_fs_high_vars[idx], width=8)
    off_fs_high.grid(row=9, column=2, sticky="w")
    tk.Label(sdm_frame, text="RS:").grid(row=10, column=0, sticky="e", padx=5)
    off_rs_low = tk.Entry(sdm_frame, textvariable=sdm_offset_rs_low_vars[idx], width=8)
    off_rs_low.grid(row=10, column=1, sticky="w")
    off_rs_high = tk.Entry(sdm_frame, textvariable=sdm_offset_rs_high_vars[idx], width=8)
    off_rs_high.grid(row=10, column=2, sticky="w")
    tk.Label(sdm_frame, text="low if pop < median − low offset, high if pop > median + high offset").grid(
        row=11, column=0, columnspan=4, sticky="w", padx=5
    )

    # --- z-score grid ---
    tk.Radiobutton(sdm_frame, text="z-score", variable=sdm_mode_vars[idx],
                   value="zscore", command=lambda i=idx: update_cl_mode(i)).grid(
        row=12, column=0, sticky="w", padx=5
    )
    tk.Label(sdm_frame, text="neg |z| (−)").grid(row=12, column=1, sticky="w")
    tk.Label(sdm_frame, text="pos |z| (+)").grid(row=12, column=2, sticky="w")
    tk.Label(sdm_frame, text="FS:").grid(row=13, column=0, sticky="e", padx=5)
    z_fs_low = tk.Entry(sdm_frame, textvariable=sdm_z_fs_low_vars[idx], width=8)
    z_fs_low.grid(row=13, column=1, sticky="w")
    z_fs_high = tk.Entry(sdm_frame, textvariable=sdm_z_fs_high_vars[idx], width=8)
    z_fs_high.grid(row=13, column=2, sticky="w")
    tk.Label(sdm_frame, text="RS:").grid(row=14, column=0, sticky="e", padx=5)
    z_rs_low = tk.Entry(sdm_frame, textvariable=sdm_z_rs_low_vars[idx], width=8)
    z_rs_low.grid(row=14, column=1, sticky="w")
    z_rs_high = tk.Entry(sdm_frame, textvariable=sdm_z_rs_high_vars[idx], width=8)
    z_rs_high.grid(row=14, column=2, sticky="w")
    tk.Label(sdm_frame, text="low if z < − neg, high if z > + pos").grid(
        row=15, column=0, columnspan=4, sticky="w", padx=5
    )

    cl_offset_entries[idx] = [off_fs_low, off_fs_high, off_rs_low, off_rs_high]
    cl_z_entries[idx] = [z_fs_low, z_fs_high, z_rs_low, z_rs_high]
    update_cl_mode(idx)

    tk.Label(sdm_frame, text="Waveform boundary (us):").grid(row=16, column=0, padx=5, pady=5)
    tk.Entry(sdm_frame, textvariable=sdm_boundary_us_vars[idx], width=10).grid(
        row=16, column=1, sticky="w"
    )
    tk.Button(
        sdm_frame, text="Build closed-loop baseline",
        command=lambda i=idx: build_closedloop_baseline(i)
    ).grid(row=17, column=0, columnspan=3, padx=5, pady=5, sticky="w")
    if sdm_vars[idx].get():
        sdm_frame.grid(row=row, column=0, columnspan=4,
                       padx=5, pady=5)
    row += 1

    # File selection subframe
    f_frame = tk.Frame(frame, borderwidth=1, relief="sunken")
    file_frames[idx] = f_frame
    tk.Label(f_frame, text="Recording binary file:").grid(
        row=0, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(f_frame, textvariable=bin_file_vars[idx], width=50).grid(
        row=0, column=1
    )
    tk.Button(
        f_frame, text="Browse", command=lambda i=idx: browse_bin_file(i)
    ).grid(row=0, column=2)
    tk.Button(
        f_frame, text="?", command=lambda i=idx: show_hint("BIN"), width=3
    ).grid(row=0, column=3)

    tk.Label(f_frame, text="Recording metadata file:").grid(
        row=1, column=0, padx=5, pady=5
    )
    tk.Entry(f_frame, textvariable=meta_file_vars[idx], width=50).grid(
        row=1, column=1
    )
    tk.Button(
        f_frame, text="Browse", command=lambda i=idx: browse_meta_file(i)
    ).grid(row=1, column=2)
    tk.Button(
        f_frame, text="?", command=lambda i=idx: show_hint("META"), width=3
    ).grid(row=1, column=3)

    tk.Label(f_frame, text="Channel map file:").grid(
        row=2, column=0, padx=5, pady=5
    )
    tk.Entry(f_frame, textvariable=chanmap_file_vars[idx], width=50).grid(
        row=2, column=1
    )
    tk.Button(
        f_frame, text="Browse", command=lambda i=idx: browse_chanmap_file(i)
    ).grid(row=2, column=2)
    tk.Button(
        f_frame, text="?", command=lambda i=idx: show_hint("CHANMAP"), width=3
    ).grid(row=2, column=3)
    if rerun_ks_vars[idx].get():
        f_frame.grid(row=row, column=0, columnspan=4,
                     padx=5, pady=5)

# Finish and error widgets
def create_finish_widgets():
    global finish_button, error_text
    finish_button = tk.Button(root, text="Finish", command=finish_and_quit)
    finish_button.grid(row=3, column=0, columnspan=4, padx=5, pady=5)
    error_text = tk.Text(root, height=4, width=60, fg="red")
    error_text.grid(row=4, column=0, columnspan=4, padx=5, pady=5)
    error_text.configure(state="disabled")

# Destroy GUI on finish
def finish_and_quit():
    root.destroy()

# Initialize GUI (with state loading and applying)
def main():
    # Load previous state
    state = None
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            num_sorters_var.set(state.get("num_sorters", num_sorters_var.get()))
            sglx_host_var.set(state.get("sglx_host", sglx_host_var.get()))
            sglx_port_var.set(state.get("sglx_port", sglx_port_var.get()))
            drift_enabled_var.set(state.get("drift_enabled", drift_enabled_var.get()))
            drift_window_var.set(state.get("drift_window_s", drift_window_var.get()))
            drift_max_shift_var.set(state.get("drift_max_shift_um", drift_max_shift_var.get()))
            drift_retrain_threshold_var.set(state.get("drift_retrain_threshold_um", drift_retrain_threshold_var.get()))
        except Exception as e:
            print(f"Could not load GUI state: {e}")

    # Build tabs
    update_tabs()

    # Apply loaded state to each tab
    if state:
        for i in range(num_sorters_var.get()):
            base_path_vars[i].set(state["base_paths"][i])
            ks_output_dir_vars[i].set(state["ks_output_dirs"][i])
            bin_file_vars[i].set(state["bin_files"][i])
            meta_file_vars[i].set(state["meta_files"][i])
            chanmap_file_vars[i].set(state["chanmap_files"][i])
            rerun_ks_vars[i].set(state["rerun_flags"][i])
            if state["rerun_flags"][i]:
                toggle_rerun(i)
            sdm_vars[i].set(state["sdm_flags"][i])
            if state["sdm_flags"][i]:
                toggle_sdm(i)
            sdm_ip_vars[i].set(state["sdm_ips"][i])
            sdm_port_vars[i].set(state["sdm_ports"][i])
            sdm_subset_vars[i].set(state.get("sdm_subsets", [""] * num_sorters_var.get())[i])
            sdm_trigger_z_vars[i].set(state.get("sdm_trigger_zs", ["1.0"] * num_sorters_var.get())[i])
            sdm_baseline_min_seconds_vars[i].set(state.get("sdm_baseline_min_seconds", ["10.0"] * num_sorters_var.get())[i])
            sdm_trigger_bin_ms_vars[i].set(state.get("sdm_trigger_bin_ms", ["50"] * num_sorters_var.get())[i])
            sdm_processor_vars[i].set(state.get("sdm_processors", ["zscore"] * num_sorters_var.get())[i])
            sdm_mode_vars[i].set(state.get("sdm_modes", ["median"] * num_sorters_var.get())[i])
            sdm_offset_vars[i].set(state.get("sdm_offsets", ["0"] * num_sorters_var.get())[i])
            n_now = num_sorters_var.get()
            legacy_off = state.get("sdm_offsets", ["0"] * n_now)[i]
            legacy_z = state.get("sdm_trigger_zs", ["1.0"] * n_now)[i]
            sdm_offset_fs_low_vars[i].set(state.get("sdm_offset_fs_low", [legacy_off] * n_now)[i])
            sdm_offset_fs_high_vars[i].set(state.get("sdm_offset_fs_high", [legacy_off] * n_now)[i])
            sdm_offset_rs_low_vars[i].set(state.get("sdm_offset_rs_low", [legacy_off] * n_now)[i])
            sdm_offset_rs_high_vars[i].set(state.get("sdm_offset_rs_high", [legacy_off] * n_now)[i])
            sdm_z_fs_low_vars[i].set(state.get("sdm_z_fs_low", [legacy_z] * n_now)[i])
            sdm_z_fs_high_vars[i].set(state.get("sdm_z_fs_high", [legacy_z] * n_now)[i])
            sdm_z_rs_low_vars[i].set(state.get("sdm_z_rs_low", [legacy_z] * n_now)[i])
            sdm_z_rs_high_vars[i].set(state.get("sdm_z_rs_high", [legacy_z] * n_now)[i])
            sdm_boundary_us_vars[i].set(state.get("sdm_boundary_us", ["200"] * num_sorters_var.get())[i])
            update_cl_mode(i)   # refresh which trigger field is live for the loaded stat
            max_templates_vars[i].set(state.get("max_templates", ["0"] * num_sorters_var.get())[i])
            channel_range_vars[i].set(state.get("channel_ranges", [""] * num_sorters_var.get())[i])

    create_finish_widgets()
    root.mainloop()

# Entry point for full run
def run_online_multi():
    main()

    # Gather inputs
    n = num_sorters_var.get()
    BASE_PATHS = [Path(v.get().strip()) for v in base_path_vars]
    KS_OUTPUT_DIRS = [Path(v.get().strip()) for v in ks_output_dir_vars]
    BIN_FILES = [Path(v.get().strip()) for v in bin_file_vars]
    META_FILES = [Path(v.get().strip()) for v in meta_file_vars]
    CHANMAP_FILES = [Path(v.get().strip()) for v in chanmap_file_vars]
    RERUN_FLAGS = [v.get() for v in rerun_ks_vars]
    SDM_FLAGS = [v.get() for v in sdm_vars]
    SDM_IPS = [v.get().strip() for v in sdm_ip_vars]
    SDM_PORTS = [v.get().strip() for v in sdm_port_vars]
    SDM_SUBSETS = [v.get().strip() for v in sdm_subset_vars]
    SDM_TRIGGER_ZS = [v.get().strip() for v in sdm_trigger_z_vars]
    SDM_BASELINE_MIN_SECONDS = [v.get().strip() for v in sdm_baseline_min_seconds_vars]
    SDM_TRIGGER_BIN_MS = [v.get().strip() for v in sdm_trigger_bin_ms_vars]
    SDM_PROCESSORS = [v.get().strip() for v in sdm_processor_vars]
    SDM_MODES = [v.get().strip() for v in sdm_mode_vars]
    SDM_OFFSETS = [v.get().strip() for v in sdm_offset_vars]
    SDM_OFFSET_FS_LOW = [v.get().strip() for v in sdm_offset_fs_low_vars]
    SDM_OFFSET_FS_HIGH = [v.get().strip() for v in sdm_offset_fs_high_vars]
    SDM_OFFSET_RS_LOW = [v.get().strip() for v in sdm_offset_rs_low_vars]
    SDM_OFFSET_RS_HIGH = [v.get().strip() for v in sdm_offset_rs_high_vars]
    SDM_Z_FS_LOW = [v.get().strip() for v in sdm_z_fs_low_vars]
    SDM_Z_FS_HIGH = [v.get().strip() for v in sdm_z_fs_high_vars]
    SDM_Z_RS_LOW = [v.get().strip() for v in sdm_z_rs_low_vars]
    SDM_Z_RS_HIGH = [v.get().strip() for v in sdm_z_rs_high_vars]
    SDM_BOUNDARY_US = [v.get().strip() for v in sdm_boundary_us_vars]
    MAX_TEMPLATES = [v.get().strip() for v in max_templates_vars]
    CHANNEL_RANGES = [v.get().strip() for v in channel_range_vars]
    SGLX_HOST = sglx_host_var.get().strip()
    SGLX_PORT = sglx_port_var.get().strip()
    DRIFT_ENABLED = drift_enabled_var.get()
    DRIFT_WINDOW_S = drift_window_var.get().strip()
    DRIFT_MAX_SHIFT_UM = drift_max_shift_var.get().strip()
    DRIFT_RETRAIN_THRESHOLD_UM = drift_retrain_threshold_var.get().strip()

    # Save current state
    state = {
        "num_sorters": n,
        "sglx_host": SGLX_HOST,
        "sglx_port": SGLX_PORT,
        "drift_enabled": DRIFT_ENABLED,
        "drift_window_s": DRIFT_WINDOW_S,
        "drift_max_shift_um": DRIFT_MAX_SHIFT_UM,
        "drift_retrain_threshold_um": DRIFT_RETRAIN_THRESHOLD_UM,
        "base_paths": [str(p) for p in BASE_PATHS],
        "ks_output_dirs": [str(d) for d in KS_OUTPUT_DIRS],
        "bin_files": [str(p) for p in BIN_FILES],
        "meta_files": [str(p) for p in META_FILES],
        "chanmap_files": [str(p) for p in CHANMAP_FILES],
        "rerun_flags": RERUN_FLAGS,
        "sdm_flags": SDM_FLAGS,
        "sdm_ips": SDM_IPS,
        "sdm_ports": SDM_PORTS,
        "sdm_subsets": SDM_SUBSETS,
        "sdm_trigger_zs": SDM_TRIGGER_ZS,
        "sdm_baseline_min_seconds": SDM_BASELINE_MIN_SECONDS,
        "sdm_trigger_bin_ms": SDM_TRIGGER_BIN_MS,
        "sdm_processors": SDM_PROCESSORS,
        "sdm_modes": SDM_MODES,
        "sdm_offsets": SDM_OFFSETS,
        "sdm_offset_fs_low": SDM_OFFSET_FS_LOW,
        "sdm_offset_fs_high": SDM_OFFSET_FS_HIGH,
        "sdm_offset_rs_low": SDM_OFFSET_RS_LOW,
        "sdm_offset_rs_high": SDM_OFFSET_RS_HIGH,
        "sdm_z_fs_low": SDM_Z_FS_LOW,
        "sdm_z_fs_high": SDM_Z_FS_HIGH,
        "sdm_z_rs_low": SDM_Z_RS_LOW,
        "sdm_z_rs_high": SDM_Z_RS_HIGH,
        "sdm_boundary_us": SDM_BOUNDARY_US,
        "max_templates": MAX_TEMPLATES,
        "channel_ranges": CHANNEL_RANGES
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Could not save GUI state: {e}")

    # Curate OSS input dirs
    OSS_DIRS = curate_oss_input_dir(BASE_PATHS, KS_OUTPUT_DIRS, BIN_FILES,
                                     META_FILES, CHANMAP_FILES, RERUN_FLAGS,
                                     MAX_TEMPLATES, CHANNEL_RANGES)

    for i, proc in enumerate(SDM_PROCESSORS):
        if SDM_FLAGS[i] and proc == 'closedloop':
            ok, msg = run_baseline_build(
                OSS_DIRS[i], SDM_BOUNDARY_US[i] or "200",
                SDM_TRIGGER_BIN_MS[i] or "100")
            tag = "[closed-loop baseline]" if ok else "[closed-loop baseline] FAILED"
            print(f"{tag} sorter {i+1}:\n{msg}")

    # Build and run C++ command
    decoder_input_dirs = [bp / 'decoder_input' for bp in BASE_PATHS]
    cuda_output_dirs = [bp / 'cuda_output' for bp in BASE_PATHS]
    for d in decoder_input_dirs + cuda_output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    arguments = {
        '--n_gpus': str(n),
        '--oss_input': OSS_DIRS,
        '--decoder_input': [str(d) + '\\' for d in decoder_input_dirs],
        '--spikes_output': [str(d / 'spikeOutput.txt') for d in decoder_input_dirs],
        '--cuda_output_dir': [str(d) + '\\' for d in cuda_output_dirs],
        '--sglx_host': SGLX_HOST,
        '--sglx_port': SGLX_PORT
    }

    if any(SDM_FLAGS):
        sdm_idxs = [i for i, enabled in enumerate(SDM_FLAGS) if enabled]
        sdm_idx = sdm_idxs[0]

        sdm_port = SDM_PORTS[sdm_idx]
        if not sdm_port.isdigit():
            print(f"Error: SDM enabled but port is not a valid integer: '{sdm_port}'")
            sys.exit(1)

        arguments['--sdm_ip'] = SDM_IPS[sdm_idx]
        arguments['--sdm_port'] = sdm_port
        arguments['--sdm_processor'] = SDM_PROCESSORS[sdm_idx] or 'zscore'

        if SDM_SUBSETS[sdm_idx]:
            arguments['--sdm_subset'] = SDM_SUBSETS[sdm_idx]
        if SDM_TRIGGER_ZS[sdm_idx]:
            arguments['--sdm_trigger_z'] = SDM_TRIGGER_ZS[sdm_idx]
        if SDM_BASELINE_MIN_SECONDS[sdm_idx]:
            arguments['--sdm_baseline_min_seconds'] = SDM_BASELINE_MIN_SECONDS[sdm_idx]
        if SDM_TRIGGER_BIN_MS[sdm_idx]:
            arguments['--sdm_trigger_bin_ms'] = SDM_TRIGGER_BIN_MS[sdm_idx]

        if SDM_PROCESSORS[sdm_idx] == 'closedloop':
            oss_dir = OSS_DIRS[sdm_idx]  # already has a trailing backslash
            arguments['--sdm_mode'] = SDM_MODES[sdm_idx] or 'median'
            arguments['--sdm_offset'] = SDM_OFFSETS[sdm_idx] or '0'

            for flag, vals in (
                ('--sdm_offset_fs_low',  SDM_OFFSET_FS_LOW),
                ('--sdm_offset_fs_high', SDM_OFFSET_FS_HIGH),
                ('--sdm_offset_rs_low',  SDM_OFFSET_RS_LOW),
                ('--sdm_offset_rs_high', SDM_OFFSET_RS_HIGH),
                ('--sdm_trigger_z_fs_low',  SDM_Z_FS_LOW),
                ('--sdm_trigger_z_fs_high', SDM_Z_FS_HIGH),
                ('--sdm_trigger_z_rs_low',  SDM_Z_RS_LOW),
                ('--sdm_trigger_z_rs_high', SDM_Z_RS_HIGH),
            ):
                if vals[sdm_idx]:
                    arguments[flag] = vals[sdm_idx]
            arguments['--sdm_rs_fs'] = oss_dir + 'rs_fs_labels.csv'
            arguments['--sdm_stats'] = oss_dir + 'closedloop_stats.txt'
            # Subset is the FS+RS neurons from the baseline, not the manual field.
            fs_rs_ids = fs_rs_ids_from_labels(oss_dir)
            if fs_rs_ids:
                arguments['--sdm_subset'] = ",".join(str(x) for x in fs_rs_ids)

    if DRIFT_ENABLED:
        if DRIFT_WINDOW_S:
            arguments['--drift_window_s'] = DRIFT_WINDOW_S
        if DRIFT_MAX_SHIFT_UM:
            arguments['--drift_max_shift_um'] = DRIFT_MAX_SHIFT_UM
        if DRIFT_RETRAIN_THRESHOLD_UM:
            arguments['--drift_retrain_threshold_um'] = DRIFT_RETRAIN_THRESHOLD_UM

    script_dir = pathlib.Path(__file__).parent.resolve()
    # Go back exactly two levels to reach the project root and build path to executable 
    exe_path = script_dir.parent.parent / "x64" / "RELEASE" / "OnlineSpikes.exe"
    # Ex: cmd = [r"C:\Users\moorelab\source\repos\LiveSpikeSorter_Remote\x64\RELEASE\OnlineSpikes.exe"]
    cmd = [str(exe_path)]
    for k, v in arguments.items():
        if isinstance(v, list):
            cmd.append(k)
            cmd.extend(v)
        else:
            cmd.extend([k, str(v)])
    cmd.append('--no_input_gui')
    if DRIFT_ENABLED:
        cmd.append('--drift_estimation')

    # RETRAIN loop
    while True:
        print("Running LSS with:", shlex.join(cmd))
        proc = subprocess.Popen(cmd, shell=True)
        rc = proc.wait()

        retrain_sources = {}   # sorter index -> (bin_path, meta_path)
        for i, oss in enumerate(OSS_DIRS):
            sentinel = Path(oss.rstrip('\\/')) / 'retrain_request.json'
            if not sentinel.exists():
                continue
            try:
                info = json.loads(sentinel.read_text())
                bin_path, meta_path = resolve_retrain_recording(info)
            except Exception as e:
                print(f"[Retrain] Could not resolve saved recording for sorter "
                      f"{i+1}: {e}. Skipping retrain for this sorter.")
                sentinel.unlink(missing_ok=True)
                continue
            retrain_sources[i] = (bin_path, meta_path)
            sentinel.unlink(missing_ok=True)

        if not retrain_sources:
            # Normal exit (user quit, or an error). rc==RETRAIN_EXIT_CODE(42)
            # without a resolvable sentinel means the recording couldn't be
            # found -- already reported above; nothing more to do.
            break

        print(f"[Retrain] Re-running Kilosort4 for sorter(s) "
              f"{[i+1 for i in retrain_sources]} on the data recorded so far. "
              f"Sorting is stopped until this finishes.")
        rerun = [(i in retrain_sources) for i in range(len(BASE_PATHS))]
        OSS_DIRS = curate_oss_input_dir(
            BASE_PATHS, KS_OUTPUT_DIRS, BIN_FILES, META_FILES, CHANMAP_FILES,
            rerun, MAX_TEMPLATES, CHANNEL_RANGES, retrain_sources=retrain_sources)
        print("[Retrain] Templates rebuilt. Relaunching sorter...")


def resolve_retrain_recording(info):
    """Given a parsed retrain_request.json, return (bin_path, meta_path) for the
    SpikeGLX file the sorter just finalized.

    The sorter records SpikeGLX's data dir, run name and probe substream, but
    the exact gate/trigger indices are awkward to reconstruct reliably. Instead
    we glob the data dir for this probe's AP files and pick the most recently
    modified one -- that is the file finalized when the button was pressed.
    """
    data_dir = info.get("data_dir", "")
    substream = int(info.get("substream", 0))
    if not data_dir or not Path(data_dir).is_dir():
        raise FileNotFoundError(f"SpikeGLX data_dir not found: {data_dir!r}")

    pattern = f"*.imec{substream}.ap.bin"
    candidates = list(Path(data_dir).rglob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No {pattern} under {data_dir} (is SpikeGLX Save enabled?)")

    bin_path = max(candidates, key=lambda p: p.stat().st_mtime)
    meta_path = bin_path.with_suffix(".meta")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta beside {bin_path.name}: {meta_path}")

    print(f"[Retrain] Using recording {bin_path} "
          f"(run_name={info.get('run_name')!r}, "
          f"file_sample_count={info.get('file_sample_count')}).")
    return str(bin_path), str(meta_path)


def cluster_centroids_pca_compute(templates, Wall, pc_feature_ind):
    """
    Compute 1D PCA‐space centroids for each template by weighting
    the per‐channel PC features in `Wall` by that channel’s
    peak-to-peak amplitude in `templates`.

    Parameters
    ----------
    templates : np.ndarray, shape (T, nt, C)
        Raw waveforms for each of T templates,
        nt time‐samples × C channels.

    Wall : np.ndarray, shape (T, C, P)
        PC‐feature “templates”: for each of the T templates,
        C channels × P PCA‐dimensions.

    pc_feature_ind : np.ndarray of ints, shape (T, K)
        For each template t, the indices of the K channels
        to use in the weighting.

    Returns
    -------
    centroids : np.ndarray, shape (T, P)
        For each template t and PCA‐dimension p,
        the weighted average of Wall[t, channel_j, p].
    """
    T, nt, C = templates.shape
    T2, C2, P = Wall.shape
    assert T == T2 and C == C2, "templates and Wall must agree on T and C"
    
    # 1) Compute peak-to-peak amplitude per template/channel: shape (T, C)
    p2p = templates.max(axis=1) - templates.min(axis=1)
    
    # 2) Allocate output
    centroids = np.zeros((T, P), dtype=Wall.dtype)
    
    # 3) Loop over templates
    for t in range(T):
        idx = pc_feature_ind[t]             # shape (K,)
        weights = p2p[t, idx].astype(float) # shape (K,)
        wsum = weights.sum()
        if wsum == 0:
            # fallback to uniform if all amplitudes are zero
            weights = np.ones_like(weights)
            wsum = float(weights.sum())
        
        # select the K×P block of PC‐features
        W_sub = Wall[t, idx, :]             # shape (K, P)
        
        # weighted average across the K channels:
        #   sum_j weights[j] * W_sub[j, :]  /  sum(weights)
        centroids[t, :] = (weights[:, None] * W_sub).sum(axis=0) / wsum
    
    return centroids

def save_kilosort_drift_plots(dshift, st0, settings):
    """
    Saves drift_amount.png into our results directory (using Kilosort4's own plotting code).
    Used for targetting training only atm

    Resulting PNGs should be identical to what Kilosort would have produced
    """
    if dshift is None or st0 is None:
        print("Drift correction disabled (nblocks=0); skipping drift plots.")
        return
    from qtpy import QtWidgets
    from kilosort.gui.sanity_plots import PlotWindow, plot_drift_amount, plot_drift_scatter
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    drift_amount_window = PlotWindow(width=400, height=400, title='Drift Amount')
    drift_scatter_window = PlotWindow(width=1500, height=700, title='Drift Scatter',
                                      background='w')
    plot_drift_amount(drift_amount_window, dshift, settings)
    plot_drift_scatter(drift_scatter_window, st0, settings)
    app.processEvents()


def run_kilosort_with_drift_plots(settings, probe_name=None, results_dir=None,
                                  filename=None, data_dtype=None, do_CAR=True,
                                  invert_sign=False, device=None):
    """
    Runs kilosort.run_kilosort() with the drift-correction outputs
    """
    import torch
    from kilosort.parameters import DEFAULT_SETTINGS
    from kilosort.run_kilosort import (
        set_files, setup_logger as ks_setup_logger, logger as ks_logger,
        initialize_ops, compute_preprocessing, compute_drift_correction,
        detect_spikes, cluster_spikes, save_sorting,
    )

    settings = {**DEFAULT_SETTINGS, **settings}
    filename, data_dir, results_dir, probe = set_files(
        settings, filename, None, probe_name, None, results_dir)
    ks_setup_logger(results_dir)

    if data_dtype is None:
        data_dtype = 'int16'
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if probe['chanMap'].max() >= settings['n_chan_bin']:
        raise ValueError(
            'Largest value of chanMap exceeds channel count of data, '
            'make sure chanMap is 0-indexed.')

    tic0 = time.time()
    ops = initialize_ops(settings, probe, data_dtype, do_CAR, invert_sign,
                         device, False)
    ops = compute_preprocessing(ops, device, tic0=tic0)

    np.random.seed(1)
    torch.cuda.manual_seed_all(1)
    torch.random.manual_seed(1)
    ops, bfile, st0 = compute_drift_correction(ops, device, tic0=tic0)

    # Snapshot the drift-stage detector before detect_spikes clobbers it.
    drift_snapshot = snapshot_drift_stage(ops, st0)

    ops['settings']['results_dir'] = str(results_dir)
    save_kilosort_drift_plots(ops.get('dshift'), st0, ops['settings'])

    st, tF, _, _ = detect_spikes(ops, results_dir, device, bfile, tic0=tic0)
    clu, Wall = cluster_spikes(results_dir, st, tF, ops, device, bfile, ks_logger,
                               tic0=tic0)
    save_sorting(ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0)

    return drift_snapshot


def _to_numpy(x, dtype=None):
    """torch tensor / list / array -> contiguous numpy array."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.ascontiguousarray(np.asarray(x))
    return x.astype(dtype) if dtype is not None else x


def snapshot_drift_stage(ops, st0, device=None):
    """
    Copy everything the live estimator needs
    """
    import torch
    from kilosort.datashift import bin_spikes, align_block2

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    F, ysamp = bin_spikes(ops, st0)
    _, _, F0, _ = align_block2(F, ysamp, ops, device=device)

    # dshift is (Nbatches, 2*nblocks-1). We estimate rigidly, so collapse the
    # blocks to one trace; with the default nblocks=1 this is a no-op.
    dshift = np.asarray(ops['dshift']).mean(axis=1)
    yc = _to_numpy(ops['yc'], np.float32)

    return {
        # universal-template detector (drift stage)
        'ud_wTEMP': _to_numpy(ops['wTEMP'], np.float32),
        'ud_iC': _to_numpy(ops['iC'], np.int32),
        'ud_iC2': _to_numpy(ops['iC2'], np.int32),
        'ud_weigh': _to_numpy(ops['weigh'], np.float32),
        'ud_ycup': _to_numpy(ops['ycup'], np.float32),
        'ud_xcup': _to_numpy(ops['xcup'], np.float32),
        # reference fingerprint + Kilosort's own training drift trace
        'ks_F0': np.ascontiguousarray(_to_numpy(F0, np.float32)),
        'ks_dshift': np.ascontiguousarray(dshift.astype(np.float32)),
        # scalars for misc.txt
        'scalars': {
            'Th_universal': float(ops['settings']['Th_universal']),
            'n_templates': int(ops['settings']['n_templates']),
            'template_sizes': int(ops['settings']['template_sizes']),
            'nearest_templates': int(ops['settings']['nearest_templates']),
            'nt': int(ops['nt']),
            'n_filters': int(_to_numpy(ops['iC']).shape[1]),
            'batch_size': int(ops['batch_size']),
            'n_amp_bins': 20,
            # yc_min/yc_max are written by the main misc.txt block; do not
            # duplicate them here (the C++ parser is a map -- last line wins).
        },
    }


def curate_oss_input_dir(BASE_PATHS, KS_OUTPUT_DIRS, BIN_FILES,
                         META_FILES, CHANMAP_FILES, RERUN_FLAGS,
                         MAX_TEMPLATES=None, CHANNEL_RANGES=None,
                         retrain_sources=None):
    retrain_sources = retrain_sources or {}
    OSS_DIRS = []
    for i, base in enumerate(BASE_PATHS):
        ks_out = KS_OUTPUT_DIRS[i]
        oss_in = base / 'oss_input'
        # Kilosort4 is checked, or on the very first curate for this sorter.
        if not RERUN_FLAGS[i] and (oss_in / 'templates.npy').exists():
            print(f"Reusing existing oss_input for sorter {i+1} "
                  "(Rerun Kilosort4 unchecked); skipping Kilosort4/torch-dependent steps.")
            OSS_DIRS.append(str(oss_in) + '\\')
            continue

        try:
            import torch
            from kilosort.preprocessing import get_drift_matrix
            from kilosort.template_matching import prepare_extract
        except ImportError as e:
            print("Error: curating oss_input/ requires the Kilosort4 training "
                  "dependencies (torch, kilosort, ...).\n"
                  "Install them with:\n"
                  "    pip install -r requirements-training.txt\n"
                  f"({e})")
            sys.exit(1)

        if RERUN_FLAGS[i]:
            if not torch.cuda.is_available():
                print(f"Error: No GPU for sorter {i+1}")
                sys.exit(1)
            if i in retrain_sources:
                bin_path, meta_path = retrain_sources[i]
                data_dir = str(Path(bin_path).parent)
                print(f"[Retrain] Starting kilosort for sorter {i+1} on {bin_path}...")
            else:
                bin_path = BIN_FILES[i]
                meta_path = META_FILES[i]
                data_dir = str(base / 'imec_raw')
                print(f"Starting kilosort for sorter {i+1}...")
            meta = parse_bin_meta_file(meta_path)
            start = time.time()
            settings = {'data_dir': data_dir, 'n_chan_bin': meta['nSavedChans']}
            drift_snapshot = run_kilosort_with_drift_plots(
                settings=settings,
                probe_name=CHANMAP_FILES[i],
                results_dir=str(ks_out),
                filename=bin_path)
            print(f"Kilosort sorter {i+1} took {time.time() - start:.2f} s")
        else:
            drift_snapshot = None

        print(f"Loading kilosort output sorter {i+1} from {ks_out}")
        # load all files
        amplitudes = np.load(ks_out / 'amplitudes.npy')
        spike_times = np.load(ks_out / 'spike_times.npy')
        spike_templates = np.load(ks_out / 'spike_templates.npy')
        spike_detection_templates = np.load(ks_out / 'spike_detection_templates.npy')
        templates      = np.load(ks_out / 'templates.npy')
        whitening_mat  = np.load(ks_out / 'whitening_mat.npy')
        channel_map    = np.load(ks_out / 'channel_map.npy')
        spike_positions = np.load(ks_out / 'spike_positions.npy')
        ops             = np.load(ks_out / 'ops.npy', allow_pickle=True).item()
        Wall3           = np.ascontiguousarray(np.load(ks_out / 'Wall3.npy'))
        ctc             = np.ascontiguousarray(np.load(ks_out / 'ctc.npy'))
        ctc_p           = torch.tensor(ctc.copy()).permute(1,0,2).contiguous().numpy()
        wPCA            = np.ascontiguousarray(np.load(ks_out / 'wPCA.npy'))
        wPCA_p          = torch.from_numpy(np.copy(wPCA)).permute(1,0).contiguous().numpy()
        centroids      = np.load(ks_out / 'cluster_centroids.npy', allow_pickle=True).item()
        centroids      = [v for v in centroids.values()]
        hpf            = np.array(ops['preprocessing']['hp_filter'])
        dshift         = np.array(ops['dshift'])
        drift_slope, _ = np.polyfit(np.arange(0, 30), dshift[-30:], 1)
        pc_feature_ind = np.load(ks_out / 'pc_feature_ind.npy')
        Wall           = np.load(ks_out / 'Wall.npy')
        cluster_centroids_pca = cluster_centroids_pca_compute(templates, Wall, pc_feature_ind)
        print(cluster_centroids_pca.shape)

        if abs(2 * drift_slope) >= 0.01: # If more than 0.01 micron drift per second in the last minute, it is unstable
            print("WARNING: Probe not yet stable. Spike sorter quality may be impacted by drift.")

        print(f"Wall3.shape = {Wall3.shape}")
        drift_matrix   = np.array(get_drift_matrix(ops, dshift[-1], device='cpu')).T
        iCC, iU, Ucc  = prepare_extract(ops, torch.tensor(Wall3), ops['settings']['nearest_chans'], device='cpu')
        iCC = np.ascontiguousarray(iCC)
        iU  = np.ascontiguousarray(iU)
        Ucc = np.ascontiguousarray(Ucc)
        xc  = np.ascontiguousarray(np.array(ops['xc']))
        yc  = np.ascontiguousarray(np.array(ops['yc']))
        pre_wf = torch.einsum('ijk,jl->kil', torch.tensor(Wall3), torch.tensor(wPCA)).permute(1,2,0).contiguous().numpy()
        print(f"Detected {channel_map.shape[0]} channels, {templates.shape[0]} templates")

        oss_in.mkdir(parents=True, exist_ok=True)
        # crop core tensors
        crop_kilosort_output(templates, whitening_mat, channel_map,
                              range(channel_map.shape[0]), range(templates.shape[0]), oss_in)
        # save all other outputs
        np.save(oss_in / 'amplitudes.npy', amplitudes)
        np.save(oss_in / 'spike_times.npy', spike_times)
        np.save(oss_in / 'spike_templates.npy',    spike_templates)
        np.save(oss_in / 'spike_detection_templates.npy', spike_detection_templates)
        np.save(oss_in / 'templates.npy',          templates)
        np.save(oss_in / 'whitening_mat.npy',      whitening_mat)
        np.save(oss_in / 'channel_map.npy',        channel_map)
        np.save(oss_in / 'spike_positions.npy',    spike_positions)
        np.savez(oss_in / 'ops.npz', **ops)
        np.save(oss_in / 'Wall3.npy', Wall3)
        np.save(oss_in / 'ctc.npy', ctc)
        np.save(oss_in / 'ctc_permuted.npy', ctc_p)
        np.save(oss_in / 'wPCA.npy', wPCA)
        np.save(oss_in / 'wPCA_permuted.npy', wPCA_p)
        np.save(oss_in / 'cluster_centroids.npy', centroids)
        np.save(oss_in / 'hp_filter.npy', hpf)
        np.save(oss_in / 'drift_matrix.npy', drift_matrix)
        np.save(oss_in / 'iCC.npy', iCC)
        np.save(oss_in / 'iU.npy', iU)
        np.save(oss_in / 'Ucc.npy', Ucc)
        np.save(oss_in / 'xc.npy', xc)
        np.save(oss_in / 'yc.npy', yc)
        np.save(oss_in / 'preclustered_template_waveforms.npy', pre_wf)
        np.save(oss_in / 'cluster_centroids_pca.npy', cluster_centroids_pca)
        # --- Real-time drift estimation exports ---
        iKxx = np.ascontiguousarray(np.array(ops['iKxx'], dtype=np.float32))
        np.save(oss_in / 'iKxx.npy', iKxx)

        binning_depth = float(ops['settings']['binning_depth'])
        sig_interp    = float(ops['settings']['sig_interp'])
        yc_min = float(np.min(yc)); yc_max = float(np.max(yc))

        # --- Universal-template detector for the LIVE drift estimator ---
        # These come from the snapshot taken mid-Kilosort, not from ops.npy /
        # ops.npz -- see snapshot_drift_stage() for why those are the wrong
        # detector. Without them the C++ estimator refuses to enable itself.
        if drift_snapshot is not None:
            for key in ('ud_wTEMP', 'ud_iC', 'ud_iC2', 'ud_weigh',
                        'ud_ycup', 'ud_xcup', 'ks_F0', 'ks_dshift'):
                np.save(oss_in / f'{key}.npy', drift_snapshot[key])
            print(f"  drift detector: {drift_snapshot['scalars']['n_filters']} "
                  f"template positions, ks_F0 {drift_snapshot['ks_F0'].shape}")
        else:
            print("  NOTE: universal-template drift exports not written "
                  "(Kilosort4 was not re-run this launch). Live drift "
                  "estimation will stay off until you re-run Kilosort4 once.")

        with open(oss_in / 'misc.txt', 'w') as f:
            f.write(f"nt0min:{ops['nt0min']}\n")
            f.write(f"numNearestChans:{ops['settings']['nearest_chans']}\n")
            f.write(f"Th_learned:{ops['Th_learned']}\n")
            f.write(f"duplicate_spike_bins:{ops['duplicate_spike_bins']}\n")
            f.write(f"sig_interp:{sig_interp}\n")
            f.write(f"binning_depth:{binning_depth}\n")
            f.write(f"yc_min:{yc_min}\n")
            f.write(f"yc_max:{yc_max}\n")
            f.write(f"dshift_last:{float(dshift[-1])}\n")
            if drift_snapshot is not None:
                for k, v in drift_snapshot['scalars'].items():
                    f.write(f"{k}:{v}\n")

        # Optionally subset preclustered templates for lower-spec GPUs and/or
        # to focus on a probe channel range. The two filters compose: the
        # channel range narrows the field first, then max_templates caps the
        # most active survivors.
        try:
            n_keep = int(MAX_TEMPLATES[i]) if MAX_TEMPLATES else 0
        except (ValueError, TypeError):
            n_keep = 0
        chan_range = parse_channel_range(CHANNEL_RANGES[i]) if CHANNEL_RANGES else None
        if n_keep > 0 or chan_range is not None:
            subset_oss_input_inplace(oss_in, n_keep, metric="spikes",
                                     allow_below_t=True, channel_range=chan_range)

        OSS_DIRS.append(str(oss_in) + '\\')
    return OSS_DIRS

if __name__ == '__main__':
    run_online_multi()
