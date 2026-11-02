"""Read checkpoint-bounded DCD segments without concatenating trajectory files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def segment_records(manifest):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    if data.get("schema") != "mecp2-production-segments-v1":
        raise ValueError("Unsupported trajectory segment manifest.")
    records = []
    previous_end = None
    interval = None
    for item in data["segments"]:
        count = int(item["safe_frames"])
        if count < 0:
            raise ValueError("A segment has a negative safe frame count.")
        if count == 0:
            continue
        path = Path(item["trajectory"])
        if not path.is_absolute():
            path = manifest.parent / path
        first = float(item["first_elapsed_ns"])
        step = float(item["interval_ns"])
        if step <= 0 or not np.isfinite([first, step]).all():
            raise ValueError("Segment time grid is invalid.")
        if interval is not None and not np.isclose(step, interval, rtol=0, atol=1e-8):
            raise ValueError("Segment output intervals differ.")
        if previous_end is not None and not np.isclose(first, previous_end + step, rtol=0, atol=1e-8):
            raise ValueError("Trajectory segments have a gap or overlap.")
        records.append(dict(path=path.resolve(), count=count, first_ns=first, interval_ns=step))
        previous_end = first + (count - 1) * step
        interval = step
    if not records:
        raise ValueError("No checkpoint-bounded trajectory frames are available.")
    return records


class SegmentReader:
    def __init__(self, path):
        from MDAnalysis.coordinates.DCD import DCDReader

        self.records = segment_records(path)
        self.readers = []
        try:
            for record in self.records:
                reader = DCDReader(str(record["path"]), convert_units=True)
                self.readers.append(reader)
                if len(reader) < record["count"]:
                    raise ValueError("A DCD is shorter than its declared safe frame count.")
            self.n_atoms = self.readers[0].n_atoms
            if any(reader.n_atoms != self.n_atoms for reader in self.readers):
                raise ValueError("Trajectory segment atom counts differ.")
            self.dt = self.records[0]["interval_ns"] * 1000
            if any(not np.isclose(reader.dt, self.dt, rtol=1e-5, atol=1e-5) for reader in self.readers):
                raise ValueError("DCD header interval differs from the segment manifest.")
            self.offsets = np.cumsum([0] + [record["count"] for record in self.records])
        except Exception:
            self.close()
            raise

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, selection):
        if isinstance(selection, slice):
            return (self[index] for index in range(*selection.indices(len(self))))
        index = int(selection)
        if not 0 <= index < len(self):
            raise IndexError(index)
        segment = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local = index - int(self.offsets[segment])
        ts = self.readers[segment][local]
        ts.time = 1000 * (self.records[segment]["first_ns"] + local * self.records[segment]["interval_ns"])
        return ts

    def close(self):
        for reader in self.readers:
            reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
