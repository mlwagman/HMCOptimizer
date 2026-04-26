#!/usr/bin/env python
"""Poll an IRI job, fetch its log on completion, parse, and log to MLflow.

Input is the JSON manifest emitted by `hmc_submit.py` (or the fields inline).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from femtomeas.workflow_manager.manager_config import readManagerConfigFile

from hmc_optimizer import _iri
from hmc_optimizer.parse import parse_log
from hmc_optimizer.propose import HMCParams
from hmc_optimizer.submit import estimate_node_hours
from hmc_optimizer.track import TrialContext, log_trial


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manager-config", required=True)
    p.add_argument("--manifest", required=True,
                   help="Path to JSON manifest emitted by hmc_submit.py")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--max-wait-seconds", type=int, default=6 * 3600)
    p.add_argument("--phase", default="refine",
                   choices=["cold_start", "refine", "production"])
    return p.parse_args(argv)


def _wait(machine: str, jobid: str, poll_s: int, max_wait_s: int) -> str:
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        state = _iri.get_job_state(machine, jobid)
        print(f"[{int(time.time()-t0):6d}s] {jobid} -> {state}")
        if state not in ("new", "queued", "active"):
            return state
        time.sleep(poll_s)
    raise TimeoutError(f"Job {jobid} did not complete in {max_wait_s}s")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    readManagerConfigFile(args.manager_config)

    manifest = json.loads(Path(args.manifest).read_text())
    machine = manifest["ctx"]["machine"]
    jobid = manifest["jobid"]
    run_dir = manifest["run_dir"]
    est_nh = float(manifest["estimated_node_hours"])
    t0 = time.time()

    final_state = _wait(machine, jobid, args.poll_seconds, args.max_wait_seconds)
    wall = time.time() - t0
    actual_nh = (manifest["ctx"]["nodes"] * wall) / 3600.0

    if final_state != "completed":
        print(f"! Job terminal state: {final_state}. Skipping MLflow log.")
        return 1

    log_path = f"{run_dir}/hmc_run.log"
    print(f"Fetching {machine}:{log_path}")
    # The IRI API doesn't expose a generic `download` yet in iri_api.py; until
    # it does, users point --manifest at a run whose log has already been
    # Globus-copied back, or set the `HMC_LOG_LOCAL` env var to a local path.
    import os
    local_log = os.environ.get("HMC_LOG_LOCAL")
    if local_log is None:
        print("! Log download not implemented; set HMC_LOG_LOCAL to a local path "
              "of the fetched stdout, or use Globus transfer.")
        return 2
    parsed = parse_log(Path(local_log).read_text())

    params = HMCParams(**manifest["params"])
    ctx = TrialContext(
        lattice=tuple(manifest["ctx"]["lattice"]),
        beta=manifest["ctx"]["beta"],
        m_l=manifest["ctx"]["m_l"],
        m_s=manifest["ctx"].get("m_s"),
        binary_path=manifest["ctx"]["binary_path"],
        binary_sha256=manifest["ctx"]["binary_sha256"],
        binary_version=manifest["ctx"]["binary_version"],
        module_list=manifest["ctx"]["module_list"],
        git_sha=manifest["ctx"]["git_sha"],
        nodes=manifest["ctx"]["nodes"],
        ranks_per_node=manifest["ctx"]["ranks_per_node"],
        machine=machine,
        account=manifest["ctx"]["account"],
        queue=manifest["ctx"]["queue"],
    )

    run_id = log_trial(
        ctx=ctx, params=params, parsed=parsed,
        estimated_node_hours=est_nh,
        actual_node_hours=actual_nh,
        proposer_tags=manifest.get("proposer_tags", {}),
        artifacts={"log": local_log},
        phase=args.phase,
    )
    print(f"Logged MLflow run: {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
