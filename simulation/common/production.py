"""Production output segments with checkpoint-bounded frame manifests."""

from __future__ import annotations

import uuid

from .checkpoints import save_snapshot, write_json


def run_production(simulation, directory, target_steps, report_interval, progress=None, log_interval=None):
    from openmm import app, unit

    if progress is None:
        progress = dict(origin_step=int(simulation.currentStep),
                        origin_time_ps=simulation.context.getState().getTime().value_in_unit(unit.picosecond),
                        segments=[])
    progress = dict(progress)
    origin = int(progress["origin_step"])
    end = origin + target_steps
    if simulation.currentStep > end:
        raise ValueError("Requested production target precedes the checkpoint.")
    segments = [dict(item) for item in progress["segments"]]
    # A checkpoint is the authoritative boundary after an interrupted write.
    safe_step = int(simulation.currentStep)
    for segment in segments:
        last = min(segment["last_step"], safe_step)
        segment["safe_frames"] = max(0, (last - segment["first_step"]) // report_interval + 1)
        segment["last_step"] = segment["first_step"] + (segment["safe_frames"] - 1) * report_interval
    progress["segments"] = segments

    def publish():
        save_snapshot(simulation, directory, "production", progress, "production")
        write_json(directory / "trajectories.json", dict(
            schema="mecp2-production-segments-v1", production_origin_step=origin,
            timestep_ps=0.01, report_interval_steps=report_interval, segments=segments))

    if simulation.currentStep == end:
        publish()
        return
    prefix = f"segment_{len(segments):04d}_{uuid.uuid4().hex[:8]}"
    first = (simulation.currentStep // report_interval + 1) * report_interval
    segment = dict(trajectory=prefix + ".dcd", state=prefix + ".csv", first_step=first,
                   last_step=first - report_interval, safe_frames=0,
                   first_elapsed_ns=(first - origin) * 0.01 / 1000,
                   interval_ns=report_interval * 0.01 / 1000)
    segments.append(segment)
    publish()

    class CheckpointReporter:
        def describeNextReport(self, sim):
            return dict(steps=report_interval - sim.currentStep % report_interval,
                        periodic=None, include=[])

        def report(self, sim, state):
            segment["last_step"] = int(sim.currentStep)
            segment["safe_frames"] = (sim.currentStep - first) // report_interval + 1
            publish()

    simulation.reporters.append(app.DCDReporter(str(directory / segment["trajectory"]), report_interval))
    simulation.reporters.append(app.StateDataReporter(str(directory / segment["state"]), log_interval or report_interval,
                                                       step=True, potentialEnergy=True,
                                                       temperature=True, volume=True))
    simulation.reporters.append(CheckpointReporter())
    print("Starting production", flush=True)
    simulation.step(end - simulation.currentStep)
    publish()
