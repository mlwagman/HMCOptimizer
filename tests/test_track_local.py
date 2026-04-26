"""Stage-2.5: validate the MLflow tracking schema against a local
file-backed tracking store. No binary, no IRI, no network.

Confirms the contract documented in SKILL.md §7: a reader running
`mlflow search_runs` sees a known set of params/metrics/tags and can
reconstruct a tuning trial from them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hmc_optimizer import _mlflow, track
from hmc_optimizer.parse import ParsedRun
from hmc_optimizer.propose import HMCParams


def _ctx() -> track.TrialContext:
    return track.TrialContext(
        lattice=(8, 8, 8, 8),
        beta=6.0,
        m_l=0.05,
        m_s=0.1,
        binary_path="/fake/hmc_driver_omelyan",
        binary_sha256="deadbeef" * 8,
        binary_version="Grid 0.7.x",
        module_list="cudatoolkit/12 PrgEnv-gnu",
        git_sha="abc1234",
        nodes=1,
        ranks_per_node=4,
        machine="Perlmutter",
        account="m4982_g",
        queue="debug",
    )


def _parsed() -> ParsedRun:
    return ParsedRun(
        plaq=0.58710,
        dH=0.04,
        accept=0.83,
        wall_per_traj_s=2.5,
        n_trajectories=10,
        cg_iters_mean=120.0,
        raw_sample_count=10,
    )


def _params() -> HMCParams:
    return HMCParams(
        integrator="omelyan", dt=0.04, md_steps=12, traj_length=0.48
    )


def test_log_trial_round_trips_through_file_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    monkeypatch.setenv("USER_EMAIL", "mlwagman@gmail.com")

    artifact = tmp_path / "fake_hmc_run.log"
    artifact.write_text("Plaquette: 0.58710\n")

    ctx = _ctx()
    run_id = track.log_trial(
        ctx=ctx,
        params=_params(),
        parsed=_parsed(),
        estimated_node_hours=0.1,
        actual_node_hours=0.09,
        proposer_tags={"strategy": "rule"},
        artifacts={"log": artifact},
        phase="cold_start",
    )
    assert isinstance(run_id, str) and run_id

    history = _mlflow.load_history(_mlflow.experiment_name(
        ctx.lattice, ctx.beta, ctx.m_l
    ))
    assert len(history) == 1
    row = history[0]
    assert row["run_id"] == run_id

    # Tags
    assert row["tags.strategy"] == "rule"
    assert row["tags.integrator"] == "omelyan"
    assert row["tags.phase"] == "cold_start"
    assert row["tags.user_email"] == "mlwagman@gmail.com"

    # Params (MLflow stores all as strings — the schema commits to that)
    assert row["params.integrator"] == "omelyan"
    assert row["params.dt"] == "0.04"
    assert row["params.md_steps"] == "12"
    assert row["params.lattice"] == "8x8x8x8"
    assert row["params.beta"] == "6.0"
    assert row["params.m_l"] == "0.05"
    assert row["params.binary_sha256"].startswith("deadbeef")
    assert row["params.git_sha"] == "abc1234"

    # Metrics
    assert float(row["metrics.plaq"]) == pytest.approx(0.58710)
    assert float(row["metrics.accept"]) == pytest.approx(0.83)
    assert float(row["metrics.dH"]) == pytest.approx(0.04)
    assert float(row["metrics.wall_per_traj_s"]) == pytest.approx(2.5)
    # wall_per_accept_s = 2.5 / 0.83
    assert float(row["metrics.wall_per_accept_s"]) == pytest.approx(2.5 / 0.83)
    assert float(row["metrics.cg_iters_mean"]) == pytest.approx(120.0)
    assert float(row["metrics.estimated_node_hours"]) == pytest.approx(0.1)
    assert float(row["metrics.actual_node_hours"]) == pytest.approx(0.09)


def test_experiment_name_canonical():
    # SKILL.md §7: experiment naming is part of the public schema.
    assert _mlflow.experiment_name((8, 8, 8, 8), 6.0, 0.05) == "hmc_tune_8x8x8x8_6_0.05"
    assert _mlflow.experiment_name((16, 16, 16, 32), 2.13, 0.01) == "hmc_tune_16x16x16x32_2.13_0.01"


def test_two_trials_share_one_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Different params, same physics point → one experiment, two runs."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    ctx = _ctx()

    common = dict(
        ctx=ctx,
        parsed=_parsed(),
        estimated_node_hours=0.1,
        actual_node_hours=0.09,
        proposer_tags={"strategy": "rule"},
    )
    track.log_trial(params=_params(), phase="cold_start", **common)
    track.log_trial(
        params=HMCParams(integrator="omelyan", dt=0.05, md_steps=10, traj_length=0.5),
        phase="refine",
        **common,
    )
    history = _mlflow.load_history(_mlflow.experiment_name(
        ctx.lattice, ctx.beta, ctx.m_l
    ))
    assert len(history) == 2
    dts = sorted(float(r["params.dt"]) for r in history)
    assert dts == [0.04, 0.05]
