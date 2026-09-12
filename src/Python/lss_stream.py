"""Receive the LSS public spike stream (OnlineSpikes.exe --spike_stream host:port).

Minimal use -- any closed-loop logic goes in the loop body:

    import lss_stream
    for batch in lss_stream.listen(port=9100):
        fs = np.isin(batch.spikes["cluster"], FS_IDS).sum()
        ...

Each item is ONE datagram. A sorter batch with more than 89 spikes arrives as
several datagrams (batch.part / batch.n_parts); each is self-contained, so most
consumers can ignore parts entirely. A batch with zero spikes still arrives, so
batch.end advances even when the recording is quiet.

Times are SpikeGLX imec stream samples; cluster is the final-cluster index (the
same IDs as column 2 of spikeOutput.txt; oss_input/templateMap.npy maps them to
Kilosort cluster IDs).

Wire format is defined in src/Networking/SpikeStream.h -- keep them in lockstep.

Smoke test (prints a one-line summary per second):
    py -3.10 src/Python/lss_stream.py --port 9100

Content check -- on Ctrl+C, compares everything received against the sorter's own
spikeOutput.txt (<base>/cuda_output/spikeOutput.txt) over the same stretch:
    py -3.10 src/Python/lss_stream.py --port 9100 --verify <path to spikeOutput.txt>
"""
import argparse
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

MAGIC = 0x3153534C  # b"LSS1"
VERSION = 1
HEADER = struct.Struct("<IHHIHHQQQIHH")  # 48 bytes, see SpikeStreamHeader
RECORD = np.dtype([("sample", "<u8"), ("cluster", "<i4"), ("amp", "<f4")])  # 16 bytes
assert HEADER.size == 48 and RECORD.itemsize == 16


@dataclass
class Batch:
    sorter: int
    seq: int          # increments once per sorter batch; gaps = dropped packets
    part: int
    n_parts: int
    start: int        # [start, end) stream samples this batch's output covers
    end: int
    sent_us: int      # sender steady_clock (QueryPerformanceCounter on Windows)
    process_us: int   # sorter compute time for the batch
    spikes: np.ndarray  # structured array: sample, cluster, amp


def parse(buf):
    """Decode one datagram. Returns None for anything that is not an LSS v1 packet."""
    if len(buf) < HEADER.size:
        return None
    (magic, version, sorter, seq, part, n_parts,
     start, end, sent_us, process_us, n, _reserved) = HEADER.unpack_from(buf)
    if magic != MAGIC or version != VERSION or len(buf) < HEADER.size + n * RECORD.itemsize:
        return None
    spikes = np.frombuffer(buf, RECORD, count=n, offset=HEADER.size)
    return Batch(sorter, seq, part, n_parts, start, end, sent_us, process_us, spikes)


def listen(port=9100, host="0.0.0.0", timeout=None):
    """Yield a Batch per datagram received on host:port. timeout (s) -> socket.timeout."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((host, port))
    sock.settimeout(timeout)
    try:
        while True:
            buf, _ = sock.recvfrom(2048)
            batch = parse(buf)
            if batch is not None:
                yield batch
    finally:
        sock.close()


def verify_against_file(spikes, path, fs=30000):
    """Compare streamed spikes with the sorter's spikeOutput.txt over the same stretch.

    Trims 1 s at each end: batches received just before/after the listener ran can
    hold spikes that sit inside the boundary samples (template alignment), which
    would read as false mismatches.
    """
    if len(spikes) == 0:
        print("[verify] No spikes received -- nothing to compare.")
        return False
    with open(path, "r") as f:
        text = f.read()
    text = text[: text.rfind("\n") + 1]   # sorter may be mid-write: drop a partial last line
    rows = np.loadtxt(text.splitlines(), delimiter=",", usecols=(0, 1, 2), ndmin=2)
    lo, hi = int(spikes["sample"].min()) + fs, int(spikes["sample"].max()) - fs
    if hi <= lo:
        print("[verify] Less than ~2 s of spikes received -- run longer before Ctrl+C.")
        return False

    s = spikes[(spikes["sample"] >= lo) & (spikes["sample"] <= hi)]
    r = rows[(rows[:, 0] >= lo) & (rows[:, 0] <= hi)]
    s_key = np.sort(s["sample"].astype(np.int64) * 100000 + s["cluster"])
    r_key = np.sort(r[:, 0].astype(np.int64) * 100000 + r[:, 1].astype(np.int64))
    only_stream = np.setdiff1d(s_key, r_key).size
    only_file = np.setdiff1d(r_key, s_key).size
    span_s = (hi - lo) / fs
    print(f"[verify] {span_s:.1f} s compared: stream {len(s_key)} spikes, file {len(r_key)} spikes, "
          f"only-in-stream {only_stream}, only-in-file {only_file}")

    ok = len(s_key) == len(r_key) and only_stream == 0 and only_file == 0
    if ok:
        # file amplitudes are printed at 6 significant digits
        s_amp = s["amp"][np.lexsort((s["cluster"], s["sample"]))]
        r_amp = r[np.lexsort((r[:, 1], r[:, 0])), 2]
        amp_ok = np.allclose(s_amp, r_amp, rtol=1e-4, atol=1e-6)
        print("[verify] MATCH: identical (sample, cluster)" + (" and amplitude." if amp_ok
              else "; amplitudes DIFFER."))
        return amp_ok
    print("[verify] MISMATCH: stream and spikeOutput.txt disagree over the same stretch.")
    return False


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--verify", metavar="SPIKEOUTPUT_TXT",
                    help="on Ctrl+C, compare everything received against this spikeOutput.txt")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)   # so Ctrl+C is honoured on Windows even when no packets arrive

    print(f"Listening on {args.host}:{args.port} ... (Ctrl+C to stop"
          f"{' and verify' if args.verify else ''})")
    last_seq = {}
    n_batches = n_spikes = n_dropped = 0
    proc_us, lat_us = [], []
    kept = []
    t_report = time.monotonic() + 1.0
    try:
        while True:
            try:
                buf, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            b = parse(buf)
            if b is None:
                continue
            # perf_counter and the sender's steady_clock both read QPC on Windows, so this is
            # transport latency on the same machine. Meaningless across machines.
            lat_us.append(time.perf_counter_ns() // 1000 - b.sent_us)
            n_spikes += len(b.spikes)
            if args.verify:
                kept.append(b.spikes.copy())
            if b.part == 0:
                n_batches += 1
                proc_us.append(b.process_us)
                prev = last_seq.get(b.sorter)
                if prev is not None and b.seq > prev + 1:
                    n_dropped += b.seq - prev - 1
                last_seq[b.sorter] = b.seq
            if time.monotonic() >= t_report:
                lat_ms = np.median(lat_us) / 1000
                # Only trust it if the two clocks evidently agree (not true on macOS, for one).
                lat_txt = f"{lat_ms:.2f} ms" if 0 <= lat_ms < 1000 else "n/a (clocks differ)"
                print(f"sorter(s) {sorted(last_seq)}  batches/s {n_batches:4d}  spikes/s {n_spikes:6d}  "
                      f"dropped (total) {n_dropped}  stream end {b.end}  "
                      f"median process {np.median(proc_us) / 1000 if proc_us else float('nan'):.1f} ms  "
                      f"median send->recv {lat_txt}")
                n_batches = n_spikes = 0
                proc_us, lat_us = [], []
                t_report = time.monotonic() + 1.0
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    if args.verify:
        if len(last_seq) > 1:
            print("[verify] Several sorters are streaming; spikeOutput.txt is per sorter. "
                  "Run with one sorter to verify.")
            return
        spikes = np.concatenate(kept) if kept else np.zeros(0, RECORD)
        verify_against_file(spikes, args.verify)


if __name__ == "__main__":
    _main()
