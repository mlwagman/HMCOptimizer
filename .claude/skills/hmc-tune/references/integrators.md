# Grid HMC integrators — deep reference

## Summary table

| Integrator | Truncation order | Cost/step | Typical dt | Notes |
|---|---|---|---|---|
| LeapFrog | O(dt²) | 1.0× | smaller | Diagnostic baseline. |
| MinimumNorm2 (Omelyan 2MN) | O(dt²), smaller coefficient | 1.0× | larger than LeapFrog at matched accept | Default. λ ≈ 0.1932. |
| ForceGradient | O(dt⁴) | ~1.33× | can be larger still | Worth trying only when dH is integrator-limited. |

Grid class names (all in `Grid/qcd/hmc/integrators/Integrator_algorithm.h`):
`LeapFrog<...>`, `MinimumNorm2<...>`, `ForceGradient<...>`. All are templates
used as `GenericHMCRunner<Integrator>` (a `HMCWrapperTemplate` alias in
`Grid/qcd/hmc/GenericHMCrunner.h`).

## Why MinimumNorm2 is a reasonable default

MinimumNorm2 is a 2-stage Omelyan-style ("2MN") integrator. It runs at the
same per-step cost as LeapFrog (two force evaluations per step) but has a
smaller error coefficient at the same truncation order (`O(dt²)`), when
λ is chosen to minimise the commutator-norm part of the shadow
Hamiltonian. In practice that lets it tolerate a somewhat larger dt at
matched acceptance. The exact speedup vs LeapFrog depends on the action
and is something to measure, not assume.

**λ sensitivity.** Grid's `MinimumNorm2` hard-codes
`lambda = 0.1931833275037836` at `Integrator_algorithm.h:143`. Values in
`[0.190, 0.196]` are ~indistinguishable for typical HMC runs; outside
that band dH grows. Don't tune λ unless dH is otherwise good and you are
trying to squeeze the last few percent.

## ForceGradient tradeoff

ForceGradient folds a second force-call-with-shift into each step,
yielding formal `O(dt⁴)` truncation. The ~1.33× per-step cost is only
amortised when:

1. dH is dominated by integrator truncation, not by stochastic force
   noise from the pseudofermion estimator.
2. CG cost per force call is small — otherwise the extra force call
   swamps any dt-doubling benefit.

Grid's comment on ForceGradient explicitly notes empirical scaling ~dH
∝ dt⁴ (`Integrator_algorithm.h:210`). If MinimumNorm2 cannot reach
`accept ∈ [0.7, 0.9]` at a reasonable dt without Hasenbusch problems,
try ForceGradient at a somewhat larger dt. Otherwise don't bother —
gate it behind `--allow-fg` so it isn't explored reflexively.

## Integrator promotion ladder used by `RuleProposer`

1. Start: MinimumNorm2 at a moderate dt (e.g. dt ~ trajL / 12).
2. If `accept < 0.6` after dt halvings down to ~0.01: stop, escalate.
3. If `accept` is in band but dH >> 1 repeatedly: try ForceGradient
   (only if `--allow-fg`), else accept the current dH.
4. LeapFrog is never proposed automatically — use it only if the user
   explicitly requests a baseline reading.

## TXQCD source references (for context, not as validated defaults)

`Grid-TXQCD/HMC/Mobius2p1f_EOFA_96I_hmc.cc` is the 96-site Möbius DWF
EOFA file in the TXQCD Grid branch. Its relevant hard-coded values:

- Integrator: `GenericHMCRunner<MinimumNorm2>` (`:181`).
- `MDsteps = 12` (`:187`).
- `trajL = 0.5` (`:188`) — note: *0.5*, not 0.48.
- `StoppingCondition = 1e-14`, `MDStoppingCondition = 1e-9` (`:283–284`).

These are the TXQCD-team's production choices at that physics point and
lattice. They are *reasonable starting points* for a fresh tuning study
on a similar ensemble; they are **not** proven optima for any other
`(L, β, m_l)`.

## Don't-fuse rules

It's tempting to change integrator and the Hasenbusch ladder in the same
trial when dH blows up. Don't — the confounding makes it impossible to
tell which change helped. Pin one, vary the other.
