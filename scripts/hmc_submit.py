#!/usr/bin/env python
"""Submit one HMC trial: propose → render → submit → log.

Typical invocation (driven by the hmc-tune skill):

    python -m scripts.hmc_submit \
        --manager-config ~/.amsc/manager.json \
        --lattice 8,8,8,8 --beta 2.13 --m-l 0.01 \
        --machine Perlmutter --account amsc013_g --queue debug \
        --nodes 1 --ranks-per-node 4 --gpus-per-rank 1 \
        --time-seconds 1800 \
        --binary /global/.../hmc_pureqcd \
        --strategy rule

See `.claude/skills/hmc-tune/SKILL.md` §8 for the full argument surface.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from femtomeas.workflow_manager.manager_config import readManagerConfigFile

from hmc_optimizer import _iri, _mlflow
from hmc_optimizer.propose import HMCParams, TrialResult, make_proposer
from hmc_optimizer.submit import (
    RenderInputs,
    binary_for,
    build_batch_script,
    build_cli_args,
    estimate_node_hours,
    submit_trial,
)
from hmc_optimizer.track import (
    TrialContext,
    binary_version_banner,
    current_git_sha,
    file_sha256,
    module_list_snapshot,
)

# Starting point only — the 9-rung ladder hard-coded in
# Grid-TXQCD/HMC/Mobius2p1f_EOFA_96I_hmc.cc:231. Not validated for other
# physics points; prune or extend per references/hasenbusch.md.
DEFAULT_HASENBUSCH = [0.005, 0.0145, 0.045, 0.108, 0.25, 0.35, 0.51, 0.6, 0.8]


def _parse_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manager-config", required=True,
                   help="JSON file consumed by femtomeas.workflow_manager.manager_config")
    p.add_argument("--lattice", required=True, type=_parse_tuple,
                   help="Lattice size as Lx,Ly,Lz,Lt")
    p.add_argument("--beta", required=True, type=float)
    p.add_argument("--m-l", required=True, type=float)
    p.add_argument("--m-s", type=float, default=None)
    p.add_argument("--machine", required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--queue", required=True)
    p.add_argument("--nodes", required=True, type=int)
    p.add_argument("--ranks-per-node", required=True, type=int)
    p.add_argument("--gpus-per-rank", type=int, default=1)
    p.add_argument("--mpi-geom", type=_parse_tuple, default=(1, 1, 1, 4))
    p.add_argument("--time-seconds", required=True, type=int)
    p.add_argument("--binary-prefix", required=True,
                   help="Absolute path up to but excluding _<integrator>, e.g. "
                        "/global/.../bin/hmc_driver; the suffix is added based "
                        "on the proposed integrator.")
    p.add_argument("--strategy", choices=["rule", "optuna"], default="rule")
    p.add_argument("--allow-fg", action="store_true")
    p.add_argument("--allow-binary-drift", action="store_true")
    p.add_argument("--n-trajectories", type=int, default=20)
    p.add_argument("--node-hours-cap", type=float, default=100.0,
                   help="Cumulative cap for this session; pre-submit confirm above this")
    p.add_argument("--dry-run", action="store_true",
                   help="Render + print, do not submit or log")
    p.add_argument("--batch-width", type=int, default=1,
                   help="Submit N candidates in parallel (queue-latency mitigation)")
    return p.parse_args(argv)


def _reconstruct_history(experiment: str, lattice) -> list[TrialResult]:
    history = _mlflow.load_history(experiment)
    out: list[TrialResult] = []
    for r in history:
        try:
            params = HMCParams(
                integrator=r["params.integrator"],
                dt=float(r["params.dt"]),
                md_steps=int(r["params.md_steps"]),
                traj_length=float(r["params.traj_length"]),
            )
            out.append(TrialResult(
                params=params,
                accept=float(r["metrics.accept"]),
                dH=float(r["metrics.dH"]),
                plaq=float(r["metrics.plaq"]),
                wall_per_traj_s=float(r["metrics.wall_per_traj_s"]),
                wall_per_accept_s=float(r["metrics.wall_per_accept_s"]),
                n_trajectories=int(float(r["metrics.n_trajectories"])),
            ))
        except (KeyError, ValueError):
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    readManagerConfigFile(args.manager_config)

    experiment = _mlflow.experiment_name(args.lattice, args.beta, args.m_l)
    history = _reconstruct_history(experiment, args.lattice)

    proposer = make_proposer(
        args.strategy,
        **({"allow_force_gradient": args.allow_fg} if args.strategy == "rule" else {}),
    )
    params = proposer.propose(history)
    if params is None:
        print(f"Converged; no further proposal. History size: {len(history)}")
        return 0

    est_nh = estimate_node_hours(
        md_steps=params.md_steps,
        n_trajectories=args.n_trajectories,
        nodes=args.nodes,
    )
    total_nh = sum(float(r.get("metrics.actual_node_hours", 0.0)) for r in
                   _mlflow.load_history(experiment))
    print(f"Proposed: {params}  estimated node-hours: {est_nh:.2f}  "
          f"cumulative: {total_nh:.2f} / cap {args.node_hours_cap:.0f}")

    if total_nh + est_nh > args.node_hours_cap:
        print("! Session node-hour cap would be exceeded. "
              "Re-run with a higher --node-hours-cap or a smaller trial.")
        return 2

    render_inputs = RenderInputs(
        params=params,
        lattice=args.lattice,
        beta=args.beta,
        m_l=args.m_l,
        m_s=args.m_s,
        hasenbusch_ladder=DEFAULT_HASENBUSCH,
        n_trajectories=args.n_trajectories,
    )
    binary_path = binary_for(args.binary_prefix, params.integrator)

    if args.dry_run:
        script = build_batch_script(
            binary_prefix=args.binary_prefix,
            integrator=params.integrator,
            cli_args=build_cli_args(render_inputs),
            lattice=args.lattice,
            mpi_geom=args.mpi_geom,
        )
        print("=== dry-run batch script ===")
        print(script)
        return 0

    sandbox = _iri.sandbox_dir(args.machine)
    run_dir = f"{sandbox}/hmc_tune/{experiment}/<JOBID>"

    sha = file_sha256(binary_path)
    ctx = TrialContext(
        lattice=args.lattice,
        beta=args.beta,
        m_l=args.m_l,
        m_s=args.m_s,
        binary_path=binary_path,
        binary_sha256=sha,
        binary_version=binary_version_banner(binary_path),
        module_list=module_list_snapshot(),
        git_sha=current_git_sha(Path(__file__).parent.parent),
        nodes=args.nodes,
        ranks_per_node=args.ranks_per_node,
        machine=args.machine,
        account=args.account,
        queue=args.queue,
    )

    prior_shas = {r.get("params.binary_sha256") for r in history}
    if prior_shas and sha not in prior_shas and not args.allow_binary_drift:
        print(f"! Binary SHA differs from prior runs in {experiment}. "
              "Pass --allow-binary-drift to override.")
        return 3

    jobid = submit_trial(
        machine=args.machine,
        account=args.account,
        queue=args.queue,
        time_seconds=args.time_seconds,
        nodes=args.nodes,
        ranks_per_node=args.ranks_per_node,
        gpus_per_rank=args.gpus_per_rank,
        inputs=render_inputs,
        binary_prefix=args.binary_prefix,
        mpi_geom=args.mpi_geom,
        run_dir=run_dir,
    )

    manifest = {
        "experiment": experiment,
        "jobid": jobid,
        "run_dir": run_dir,
        "params": params.__dict__,
        "estimated_node_hours": est_nh,
        "ctx": {k: v for k, v in ctx.__dict__.items()},
        "proposer_tags": proposer.mlflow_tags(),
    }
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
