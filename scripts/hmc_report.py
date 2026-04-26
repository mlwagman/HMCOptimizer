#!/usr/bin/env python
"""Summarise an HMC tuning experiment and print the next proposal.

Reads MLflow history for the given `(lattice, beta, m_l)`, shows the best run
so far, prints convergence status per §2 of the SKILL.md decision flow, and
emits what the default RuleProposer would pick next.
"""
from __future__ import annotations

import argparse
import sys

from hmc_optimizer import _mlflow
from hmc_optimizer.propose import HMCParams, RuleProposer, TrialResult


def _parse_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lattice", required=True, type=_parse_tuple)
    p.add_argument("--beta", required=True, type=float)
    p.add_argument("--m-l", required=True, type=float)
    p.add_argument("--allow-fg", action="store_true")
    args = p.parse_args(argv or sys.argv[1:])

    experiment = _mlflow.experiment_name(args.lattice, args.beta, args.m_l)
    runs = _mlflow.load_history(experiment)
    print(f"Experiment: {experiment}  runs: {len(runs)}")

    if not runs:
        print("No runs yet. RuleProposer cold-start would be:")
        print(RuleProposer().propose([]))
        return 0

    history: list[TrialResult] = []
    for r in runs:
        try:
            history.append(TrialResult(
                params=HMCParams(
                    integrator=r["params.integrator"],
                    dt=float(r["params.dt"]),
                    md_steps=int(r["params.md_steps"]),
                    traj_length=float(r["params.traj_length"]),
                ),
                accept=float(r["metrics.accept"]),
                dH=float(r["metrics.dH"]),
                plaq=float(r["metrics.plaq"]),
                wall_per_traj_s=float(r["metrics.wall_per_traj_s"]),
                wall_per_accept_s=float(r["metrics.wall_per_accept_s"]),
                n_trajectories=int(float(r["metrics.n_trajectories"])),
            ))
        except (KeyError, ValueError):
            continue

    best = min(history, key=lambda h: h.wall_per_accept_s)
    print("\nBest so far:")
    print(f"  {best.params}")
    print(f"  accept={best.accept:.3f}  dH={best.dH:.3f}  "
          f"wall/accept={best.wall_per_accept_s:.1f}s  N={best.n_trajectories}")

    print("\nNext proposal (rule):")
    nxt = RuleProposer(allow_force_gradient=args.allow_fg).propose(history)
    print("  CONVERGED" if nxt is None else f"  {nxt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
