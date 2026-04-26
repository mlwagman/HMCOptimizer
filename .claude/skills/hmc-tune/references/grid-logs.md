# Parsing Grid HMC logs

Grid's stdout format depends on what the top-level driver prints.
`parse.py` extracts the fields that the core HMC loop
(`Grid/qcd/hmc/HMC.h`) emits directly, plus optional fields that
observable-level code typically adds. If runs suddenly parse as "no
trajectories found", check the patterns below against a raw log before
assuming the parser is at fault.

## Fields emitted by the HMC core

From reading `Grid-TXQCD/Grid/qcd/hmc/HMC.h`:

| Field | Message form | Notes |
|---|---|---|
| Trajectory number | `-- # Trajectory = N` | `:261` |
| dH | `Total H after trajectory = ... dH = <value>` | `:230` (line emits both total H and dH) |
| Metropolis outcome | `Metropolis_test -- ACCEPTED` / `-- REJECTED` | `:128`, `:132` |
| Wall time | `Total time for trajectory (s): <seconds>` | `:283` |

`Plaquette` is **not** printed by the HMC core. If you need it per
trajectory, wire up a plaquette-observable in the driver `.cc` (the
TXQCD drivers do this explicitly). The same applies to `ConjugateGradient
converged in N iterations` lines — those come from CG action code, only
if CG verbosity is high enough.

## Regex set

The exact regexes `parse.py` uses live in
`src/hmc_optimizer/parse.py`; they are derived from the message forms
above. If a newer Grid release changes any of these strings, update the
regex and note the new form here.

## Log-verbosity flag

Grid's `--log` flag takes a comma-separated list of tags, parsed in
`Grid-TXQCD/Grid/util/Init.cc:520`. Valid tags include:
`Error,Warning,Message,Performance,Iterative,Integrator,Debug,Colours`.
Trajectory / dH / Metropolis lines are at the `Message` level and are
on by default. CG iteration counts need `Iterative` or `Performance`
turned on.

There is no `--performance` flag; use `--log Performance,Message,...`
instead.

## Sanity checks after parsing

- `plaq` (when present) should be within about 1% of the target for
  your β. A large offset indicates a config-loading bug; stop and
  inspect before logging.
- `dH` standard deviation across trajectories much larger than the
  mean indicates the trial isn't thermalised yet; skip the first few
  trajectories with `parse.tail_stats(..., tail=20)`.
- If `accept` differs from `exp(-dH)` by much more than statistical
  noise, the MD integration may not be reversible — check for silent
  CG non-convergence.

## Reversibility check

Grid has reversibility logic in `HMC.h:217–224` but it is guarded by
`if(0)` in the current source — there is no standalone
`Test_hmc_reversibility` test. To check reversibility, either
un-comment that block in a private Grid build, or run the same
trajectory twice from the same checkpoint with opposite momentum sign
and compare the final plaquette.

## Log retrieval in the current IRI client

`iri_api.py` does not yet expose a generic download. Options:

1. **Globus transfer** the run directory back using
   `_iri.globus_copy_from_machine`.
2. **Set `HMC_LOG_LOCAL`** to a pre-fetched local copy before running
   `hmc_status.py`.

The second is the quick path for demo sessions; the first is correct
for production.
