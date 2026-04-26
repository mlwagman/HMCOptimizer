"""Extract HMC metrics from a Grid stdout log.

Patterns here target the message forms emitted by `Grid/qcd/hmc/HMC.h`
in the TXQCD-bundled Grid branch:

    -- # Trajectory = N
    Total H after trajectory    =   ...    dH =   <value>
    Metropolis_test -- ACCEPTED    (or -- REJECTED)
    Total time for trajectory (s): <seconds>

`Plaquette:` and `ConjugateGradient converged in N iterations` are **not**
emitted by the HMC core — they only appear when the top-level driver
explicitly instantiates a plaquette observable or raises CG verbosity.
We match them optionally.

If parsing fails after a Grid update, see
`.claude/skills/hmc-tune/references/grid-logs.md`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class ParsedRun:
    plaq: float
    dH: float
    accept: float
    wall_per_traj_s: float
    n_trajectories: int
    cg_iters_mean: float | None
    raw_sample_count: int


# HMC core output.
_RE_DH = re.compile(
    r"Total\s+H\s+after\s+trajectory\s*=\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    r"\s*dH\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_RE_ACCEPT_FLAG = re.compile(
    r"Metropolis_test\s*--\s*(ACCEPTED|REJECTED)", re.IGNORECASE
)
_RE_TRAJ_TIME = re.compile(
    r"Total\s+time\s+for\s+trajectory\s*\(s\)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Driver-level optional fields.
# Grid's `PlaquetteLogger` emits `Plaquette: [ <traj> ] <value>` (see
# Grid/qcd/observables/plaquette.h). The bracketed trajectory prefix is
# skipped here so only the value is captured.
_RE_PLAQ = re.compile(
    r"[Pp]laquette\s*[:=]\s*(?:\[\s*\d+\s*\]\s*)?"
    r"([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)"
)
_RE_CG_ITERS = re.compile(
    r"ConjugateGradient\s+(?:converged\s+in\s+|Converged[: ]+)(\d+)\s+iterations?",
    re.IGNORECASE,
)


def parse_log(text: str) -> ParsedRun:
    """Parse a full Grid HMC stdout into a `ParsedRun`.

    Raises `ValueError` if no Metropolis accept/reject lines are found.
    """
    dHs = [float(m) for m in _RE_DH.findall(text)]
    accepts = [m.upper() == "ACCEPTED" for m in _RE_ACCEPT_FLAG.findall(text)]
    traj_times = [float(m) for m in _RE_TRAJ_TIME.findall(text)]
    plaqs = [float(m) for m in _RE_PLAQ.findall(text)]
    cg_iters = [int(m) for m in _RE_CG_ITERS.findall(text)]

    if not accepts:
        raise ValueError(
            "No Metropolis_test lines found — Grid log format may have "
            "drifted. See references/grid-logs.md for the expected forms."
        )

    return ParsedRun(
        plaq=mean(plaqs) if plaqs else float("nan"),
        dH=mean(dHs) if dHs else float("nan"),
        accept=sum(accepts) / len(accepts),
        wall_per_traj_s=mean(traj_times) if traj_times else float("nan"),
        n_trajectories=len(accepts),
        cg_iters_mean=mean(cg_iters) if cg_iters else None,
        raw_sample_count=len(plaqs),
    )


def parse_log_file(path: str | Path) -> ParsedRun:
    return parse_log(Path(path).read_text())


def wall_per_accept(run: ParsedRun) -> float:
    """Derived metric: wall-clock seconds per *accepted* trajectory."""
    if run.accept <= 0:
        return float("inf")
    return run.wall_per_traj_s / run.accept


def tail_stats(texts: Sequence[str], tail: int = 20) -> ParsedRun:
    """Parse the last `tail` trajectories from one or more log chunks.

    Useful for skipping thermalisation when deciding whether a run has
    converged on its steady-state acceptance.
    """
    combined = "\n".join(texts)
    full = parse_log(combined)
    if full.n_trajectories <= tail:
        return full
    accepts_bool = [
        m.upper() == "ACCEPTED" for m in _RE_ACCEPT_FLAG.findall(combined)
    ]
    traj_times = [float(m) for m in _RE_TRAJ_TIME.findall(combined)]
    accepts_tail = accepts_bool[-tail:]
    times_tail = traj_times[-tail:] if traj_times else []
    return ParsedRun(
        plaq=full.plaq,
        dH=full.dH,
        accept=sum(accepts_tail) / len(accepts_tail),
        wall_per_traj_s=mean(times_tail) if times_tail else full.wall_per_traj_s,
        n_trajectories=len(accepts_tail),
        cg_iters_mean=full.cg_iters_mean,
        raw_sample_count=full.raw_sample_count,
    )
