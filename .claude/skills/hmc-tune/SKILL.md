---
name: hmc-tune
description: >-
  Tune Grid HMC hyperparameters (integrator, dt, MDsteps, trajectory
  length, Hasenbusch mass ladder) for QCD gauge generation on AmSC
  compute via the IRI API, with MLflow tracking. Covers LeapFrog /
  ForceGradient / MinimumNorm2 (Omelyan) integrators, targets
  acceptance ∈ [0.7, 0.9] with dH < 1.0, and minimises wall-time per
  accepted trajectory. Use this skill when the user asks to tune
  gauge generation, diagnose bad acceptance / large dH / plaquette
  drift, or extend an existing HMC tuning study. Not for Hadrons
  measurements (see HadronsJobBuilder), not for Grid builds, not for
  TXQCD composite actions (FTHMC etc.).
---

# HMC hyperparameter tuning

Adaptive tuning of Grid HMC for QCD gauge generation on AmSC. Each
invocation decides whether to cold-start, continue, or stop an
existing experiment; submits a single trial (or a narrow parallel
batch); parses Grid's log; records the result to MLflow; and reports
the next proposal.

## 1. What this skill does

**In scope.** QCD gauge generation with Grid HMC — pure gauge
(Yang-Mills) or dynamical-fermion ensembles (Nf=2, 2+1f, DWF / Möbius
DWF variants). Integrators: LeapFrog, MinimumNorm2 (Omelyan),
ForceGradient. Hyperparameters: `dt`, `MDsteps`, trajectory length,
Hasenbusch mass ladder (when fermions are present).

**Out of scope.** Hadrons measurement workflows (that's
HadronsJobBuilder). TXQCD composite actions / FTHMC / flow-based
preconditioners. Building Grid or the HMC binary — the user supplies
a path. Gauge-action tuning (β fixed by physics). Autocorrelation
analysis of produced gauge fields (use a separate observable-level
tool).

**Required inputs (from the user or the conversation).**

- Lattice size `(Lx, Ly, Lz, Lt)`.
- Physics point: β, m_l, and (if 2+1f) m_s.
- Pre-built Grid HMC binary path on the target machine.
- AmSC manager config JSON (IRI + sandbox + machine map).
- Compute envelope: machine, account, queue, nodes, ranks-per-node, time
  budget.

**Produced outputs.**

- One MLflow run per trial, with the full schema in §7.
- A JSON manifest per submission (from `scripts/hmc_submit.py`) — the
  durable state the driver reads back.
- A final recommendation: `(integrator, dt, MDsteps, traj_length)` plus the
  MLflow run URL of the winning trial.

## 2. Decision flow

On invocation, classify the request then follow the matching branch.

**A. Cold start** — no prior runs for this `(lattice, β, m_l)`.

1. Check MLflow: `hmc_report.py --lattice ... --beta ... --m-l ...`. If it
   reports `runs: 0`, proceed; otherwise treat as "extend".
2. Submit trial 0 with `RuleProposer` defaults: MinimumNorm2,
   MDsteps=12, trajL=0.5 (matching the TXQCD `Mobius2p1f_EOFA_96I_hmc.cc`
   values at `:187–188`; dt ≈ trajL / MDsteps ≈ 0.042).
3. Wait for completion (see §8), parse, log to MLflow, recurse.

**B. Extend** — runs exist, user wants more tuning.

1. Load history from MLflow.
2. Check stopping criterion: `accept ∈ [0.70, 0.90]` ∧ `dH < 1.0` ∧ the two
   most recent accepting trials agree within 10% on `wall_per_accept_s`.
   If all three hold, stop and report the winner. Otherwise, propose next.
3. Show the user the proposed params and the estimated node-hours before
   submitting (§10).

**C. Diagnose** — user reports a symptom ("dH blowing up", "low
acceptance"). Map to action per §9 *without* submitting a new trial until
the user confirms the diagnosis.

**Hard stopping criterion (applies to all branches).**

    accept ∈ [0.70, 0.90]
    AND dH < 1.0
    AND |wpa(-1) - wpa(-2)| / wpa(-2) < 0.10

where `wpa = wall_per_accept_s` over accepting trials only. If none of the
last 10 trials satisfy the acceptance+dH constraints, raise to the user —
the problem is structural, not in `dt`.

**Escape hatches.** Never issue more than 3 trials without explicit user
confirmation (§10). Never propose dt < 0.005 — that's the `IntractableError`
floor in `propose.py`. Never change integrator *and* Hasenbusch ladder in
the same trial.

## 3. Integrator cheat sheet

| Integrator | Truncation order | Cost/step | Default dt | When |
|---|---|---|---|---|
| LeapFrog | O(dt²) | 1.0× | smaller | Diagnostic baseline only |
| MinimumNorm2 (Omelyan 2MN, λ≈0.1932) | O(dt²), smaller coefficient | 1.0× | moderate | **Default** |
| ForceGradient | O(dt⁴) | ~1.33× | larger | Integrator-limited dH, gate `--allow-fg` |

Why MinimumNorm2 first: same per-step cost as LeapFrog and the
λ-tuned commutator term shrinks the error coefficient, letting it
tolerate a larger dt at matched acceptance. The exact speedup vs
LeapFrog is problem-dependent — measure, don't assume. Deep reference:
`references/integrators.md`.

## 4. Hasenbusch ladder

Starting point: `{0.005, 0.0145, 0.045, 0.108, 0.25, 0.35, 0.51, 0.6, 0.8}`
— the 9-rung ladder hard-coded in
`Grid-TXQCD/HMC/Mobius2p1f_EOFA_96I_hmc.cc:231`. Not validated for other
physics points; treat as a starting point for tuning.

Rules of thumb:

- Smaller lattice / lighter `m_l` → drop rungs from the *heavy* end first.
- 6–9 rungs is typical; more rungs cost CG time without force benefit.
- Don't touch the ladder and the integrator in the same trial.
- Pure gauge (Yang-Mills) has no fermion determinants → pass `[]`.

Deep reference: `references/hasenbusch.md`.

## 5. Parsing Grid logs

`hmc_optimizer.parse.parse_log` extracts `(dH, accept, wall_per_traj_s)`
and — if the driver prints them — `plaq` and CG iteration counts.
`plaq` and CG-iteration lines are **not** emitted by Grid's HMC core;
they come from plaquette observables and CG verbosity in the driver
`.cc`. If the parser returns zero trajectories, check
`references/grid-logs.md` against a raw log — Grid's wording
(`Metropolis_test -- ACCEPTED`, `Total H after trajectory = ... dH = ...`,
`Total time for trajectory (s):`) can drift between releases.

**Quick sanity.** When `plaq` is available, it should be within about
1% of the β-expected value; a larger gap indicates a config-loading bug,
not a tuning problem. Stop.

## 6. Driver & binary invocation

One C++ source (`templates/driver/hmc_driver.cc`), built three times —
once per integrator — via `templates/driver/Makefile` against the user's
pre-built Grid install on Perlmutter. The integrator is selected at
compile time with `-DHMC_INTEGRATOR=LeapFrog|MinimumNorm2|ForceGradient`;
the Makefile produces:

- `hmc_driver_leapfrog`
- `hmc_driver_omelyan` (MinimumNorm2, default)
- `hmc_driver_fg` (ForceGradient)

Every other hyperparameter is a **runtime CLI flag**, so one build serves
an entire tuning sweep. `hmc_optimizer.submit.build_cli_args` assembles
the argv, `build_batch_script` composes the `srun` invocation, and
`submit_trial` ships it via IRI — no per-trial source upload.

Canonical flag surface (see the driver source for full list):

    --grid Lx.Ly.Lz.Lt --mpi Mx.My.Mz.Mt        # Grid's own parser
    --beta --u0 --m-light --m-strange --csw
    --stout-rho --stout-nsmear
    --md-steps --traj-length
    --cg-tol --cg-max
    --hasenbusch m1,m2,...  --hasenbusch-top M
    --rat-lo --rat-hi --rat-degree --rat-precision
    --n-trajectories --no-metropolis-until --start-trajectory
    --starting-type {ColdStart,HotStart,TepidStart,CheckpointStart,...}
    --ckpt-prefix --ckpt-interval

Action content: 2+1 flavour Wilson-Clover + Symanzik-improved gauge +
stout-smeared fermions; light quarks via Hasenbusch-preconditioned Nf=2
ratios (empty `--hasenbusch` degenerates to a single plain action at
m_light), strange via Nf=1 rational pseudofermion.

CG tolerances: **action 1e-14, MD 1e-9** (override via `--cg-tol`). Don't
tune these unless dH is otherwise good — a tighter MD solver rarely
changes acceptance but substantially increases cost.

The user supplies `--binary-prefix` to `hmc_submit.py`; the `_leapfrog`,
`_omelyan`, or `_fg` suffix is appended based on the proposed integrator.

## 7. MLflow conventions

Experiment name: `hmc_tune_{lattice}_{beta}_{m_l}` (e.g.
`hmc_tune_8x8x8x8_2.13_0.01`).

**Params** (stringified, so `mlflow.search_runs` filters work):

- Hyperparameters: `integrator`, `dt`, `md_steps`, `traj_length`.
- Physics: `beta`, `m_l`, `m_s`, `lattice`.
- Compute: `machine`, `account`, `queue`, `nodes`, `ranks_per_node`.
- Reproducibility: `git_sha`, `binary_path`, `binary_sha256`,
  `binary_version` (captured `--version` banner), `module_list`.

**Metrics** (float):

- Physics: `plaq`, `dH`, `accept`.
- Performance: `wall_per_traj_s`, `wall_per_accept_s`, `n_trajectories`,
  `cg_iters_mean`.
- Cost accounting: `estimated_node_hours`, `actual_node_hours`.

**Tags**:

- `phase ∈ {cold_start, refine, production}`.
- `strategy ∈ {rule, optuna}` (+ `optuna_study` when applicable).
- `integrator` (mirrors the param for easy UI filtering).
- `user_email`.

**Artifacts.** The rendered `.cc`, the full Grid stdout, and — on promotion
to `phase=production` — the Globus task ID for the checkpoint stage-out.

## 8. Service integration quickref

- **IRI submit**: `hmc_optimizer._iri.submit_batch_job(...)` — the
  narrow alias that isolates the import from `femtomeas.workflow_manager`.
  All HMC modules import from `_iri`, never from `femtomeas` directly.
- **Job polling**: `_iri.get_job_state(machine, jobid)` returns one of
  `new, queued, active, completed, cancelled, failed`.
- **Globus**: only on `phase=production`. Per-trial checkpoints stay on the
  machine to avoid DTN traffic during cold-start and refine phases.
- **MLflow**: local tracking server on `http://localhost:5000` for demos;
  `MLFLOW_TRACKING_URI` overrides for a remote tracking server.
- **401 / auth failures**: one retry is automatic via
  `_iri.with_retry_on_auth`. On second failure, see `references/auth.md`.

Queue-latency mitigation. Pass `--batch-width 3` to `hmc_submit.py` and the
proposer emits three nearby candidates (dt/2, dt, dt×1.3); the skill
consumes whichever completes first, cancels the siblings, and tags the
cancelled runs with `cancelled=true` in MLflow.

## 9. Common failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| `accept < 0.3` and `dH >> 1` | dt too large for integrator | Halve dt; single-trial reaction. |
| `accept > 0.97` | dt too small (inefficient) | Grow dt ×1.3. |
| `accept ∈ [0.7, 0.9]` but `dH > 1.5` variance | Force spikes from a Hasenbusch rung | Check `references/hasenbusch.md`; add a rung, don't touch dt. |
| `plaq` off β-expected by >1% | Config-loading bug | **Stop.** Do not log trial. Investigate binary + checkpoint. |
| Parser returns 0 trajectories | Grid log format drift | Update regex set per `references/grid-logs.md`. |
| Same run produces different results | Integrator non-reversible (silent CG non-convergence) | Raise `cg_max_iter`; enable driver-level CG verbosity (`--log Iterative`) and confirm convergence lines appear. |
| MLflow 401 | Token expired | `with_retry_on_auth` retries once; on failure re-issue MLflow token. |
| IRI 401 | Globus token expired | Same — retry happens automatically; else re-run device-code flow. |
| Job stuck "queued" > 6h | Queue / account problem | Cancel, switch queue. Not a tuning issue. |
| Parser returns `accept=0` | All rejected — usually dH catastrophic | Halve dt. If persists at dt=0.01, escalate integrator. |

**Reversibility test.** When in doubt, run a trajectory forward then
its exact time-reverse. `plaq_before ≈ plaq_after` within 1e-10
confirms MD is reversible. Grid has a reversibility code path in
`Grid/qcd/hmc/HMC.h` guarded by `if(0)` — there is no standalone
`Test_hmc_reversibility` binary; to use this test, uncomment that
block in a private Grid build, or run the same trajectory twice from
the same checkpoint with opposite momentum sign.

## 10. Cost discipline

- **Hard rule: ≤3 trials without explicit user confirmation.** After three,
  summarise state and ask before continuing.
- **Per-trial wall budget: 30 minutes default.** Override with
  `--time-seconds`.
- **Pre-submit display:** always show `estimated_node_hours`, current
  cumulative total, and the configured cap (default 100 node-hours)
  *before* submitting.
- **Cap enforcement:** `scripts/hmc_submit.py` refuses submission if
  `cumulative + estimated > cap` and prints a clear remediation message.
- **Estimated vs actual:** both are logged as MLflow metrics. If the
  estimate is off by >30% consistently, update
  `submit.estimate_node_hours(est_sec_per_md_step=...)`.

## 11. References index

| File | Content |
|---|---|
| `references/integrators.md` | LeapFrog / FG / MinimumNorm2 tradeoffs, λ sensitivity, promotion ladder. |
| `references/hasenbusch.md` | Ladder choice, rung dropping and adding, tuning order. |
| `references/grid-logs.md` | Regex set, Grid version gotchas, log-retrieval paths. |
| `references/auth.md` | IRI + MLflow + Globus token flow, resume procedure. |

| CLI | Purpose |
|---|---|
| `scripts/hmc_submit.py` | Propose → render → submit → emit manifest. |
| `scripts/hmc_status.py` | Poll IRI, fetch log, parse, log to MLflow. |
| `scripts/hmc_report.py` | Summarise experiment, print next proposal. |
