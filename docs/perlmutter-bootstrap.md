# Perlmutter bootstrap for HMCOptimizer

**Audience.** A Claude Code agent (or the human user) sitting in
`$SCRATCH/Grid-TXQCD` on Perlmutter under the NERSC user `mlwagman`, with a
working Grid-TXQCD build already present.

**Goal.** Stand up HMCOptimizer next to that Grid build, run the local
integration test against a freshly compiled driver, then drive a real
single-trial HMC submission through the AmSC IRI API.

The instructions below are **idempotent** — re-running them after a crash
or a relogin should be safe.

---

## 0. Orient yourself

```bash
echo "user:        $USER"          # expect: mlwagman
echo "working in:  $PWD"           # expect: .../scratch/.../Grid-TXQCD
ls                                  # expect: build/, Grid/, ...
```

If `$PWD` is not under `$SCRATCH`, stop and ask. Everything below assumes
`$SCRATCH/Grid-TXQCD` is the *parent of the Grid build tree*, not the build
tree itself.

Locate `grid-config`:

```bash
GRID_CONFIG=$(find "$PWD" -maxdepth 4 -name grid-config -perm -u+x | head -1)
echo "$GRID_CONFIG"
"$GRID_CONFIG" --cxx --cxxflags --ldflags --libs
```

If nothing is found, the Grid build is not yet installed; bail out and ask
the human to point you at it.

## 1. Clone the two AmSC repos as siblings of Grid-TXQCD

HMCOptimizer expects HadronsJobBuilder to live next to it (the
`pyproject.toml` line `"femtomeas @ file://../HadronsJobBuilder"` resolves
relatively).

```bash
cd "$SCRATCH"
mkdir -p AmSC && cd AmSC
[ -d HadronsJobBuilder ] || git clone https://github.com/FemtoMind/HadronsJobBuilder.git
[ -d HMCOptimizer ]      || git clone https://github.com/mlwagman/HMCOptimizer.git
ls    # expect: HadronsJobBuilder/  HMCOptimizer/
```

We deliberately put these in `$SCRATCH/AmSC/`, not under `Grid-TXQCD/`, so
they survive a Grid rebuild.

## 2. Set up a Python venv

Perlmutter's stock Python is fine; pin to 3.11+:

```bash
module load python                   # currently gives a Conda Python ≥3.11
python3 -m venv "$SCRATCH/AmSC/.venv"
source "$SCRATCH/AmSC/.venv/bin/activate"
python -V                            # sanity: ≥ 3.11
pip install --upgrade pip
```

Add a one-liner to `~/.bashrc` (or just remember it) so future sessions
re-enter the venv:

```bash
# ~/.bashrc
alias hmc-env='module load python && source $SCRATCH/AmSC/.venv/bin/activate'
```

## 3. Install both packages editable

```bash
cd "$SCRATCH/AmSC/HMCOptimizer"
pip install -e ../HadronsJobBuilder    # provides femtomeas (IRI client)
pip install -e .[dev]                  # installs hmc-optimizer + pytest

# smoke check: imports + console scripts
python -c "import hmc_optimizer, femtomeas; print('ok')"
hmc-submit --help | head -5
```

If the `femtomeas` install fails, it's almost always a missing
optional dep — re-run with `pip install -e ../HadronsJobBuilder --verbose`
and read the actual error before guessing.

## 4. Build the three HMC driver binaries

The driver source lives at
`.claude/skills/hmc-tune/templates/driver/hmc_driver.cc`. The Makefile
compiles it three times, once per integrator macro, against whatever
`grid-config` you point it at.

```bash
cd "$SCRATCH/AmSC/HMCOptimizer/.claude/skills/hmc-tune/templates/driver"

# Load the EXACT module set used to build Grid-TXQCD. If you don't know it,
# look at the Grid build log (build/config.log) or ask the human.
module load PrgEnv-gnu cudatoolkit cray-mpich   # adjust to match Grid build

make GRID_CONFIG="$GRID_CONFIG" -j4

ls -la hmc_driver_leapfrog hmc_driver_omelyan hmc_driver_fg
```

Three executables, one per integrator. They are gitignored — every host
rebuilds.

If the link step fails on `-lGrid` not found, your `module load` set
doesn't match the one Grid was configured with. Re-source whatever
`sourceme.sh` lives in the Grid build tree before retrying.

## 5. Run the local integration test

This exercises the full pipeline (CLI args → subprocess → log parse →
proposer) against the just-built `omelyan` binary. No IRI, no allocation
required — runs on the login node in a couple of minutes.

```bash
cd "$SCRATCH/AmSC/HMCOptimizer"
export HMC_DRIVER_PREFIX="$SCRATCH/AmSC/HMCOptimizer/.claude/skills/hmc-tune/templates/driver/hmc_driver"
pytest -m slow tests/test_integration_local.py -v
```

Expected: 2 tests pass. If they don't, **stop and read the log** at
`tmp_path/hmc_run.log` — the trail of CG residuals, dH, and `Plaquette:`
lines tells you immediately whether it's a build-mismatch (assertion
abort), a tuning issue (CG NaN), or a parser drift (Grid output format
changed).

> Note: login nodes on Perlmutter forbid GPU work. The integration test
> runs CPU-only (4⁴ lattice, single rank). For anything bigger, allocate a
> compute node — see Stage 3.

## 6. Stage 3 — first real IRI submission

This is the actual goal: submit a single 8⁴ tuning trial to Perlmutter via
the AmSC IRI API and round-trip the result through MLflow.

### 6a. Auth setup (one-time per user)

You need:
- A NERSC Superfacility client key (PEM file). Generate at
  <https://iris.nersc.gov/profile> → "Superfacility API". Save to
  `$HOME/.superfacility/clientkey.pem` and `chmod 600` it.
- An IRI Globus auth token. The first IRI call triggers a device-code
  flow — Claude will print a URL; the human opens it, copies the code
  back. Tokens persist to a path you specify.

Create the manager config (`pyproject.toml` references it):

```bash
mkdir -p "$HOME/.hmc_optimizer"
cat > "$HOME/.hmc_optimizer/manager_config.json" <<'JSON'
{
  "workflow": {
    "sfapi_key_path":  "/global/homes/m/mlwagman/.superfacility/clientkey.pem",
    "iriapi_key_path": "/global/homes/m/mlwagman/.iri/tokens.json",
    "sandbox_directories": {
      "Perlmutter": "/pscratch/sd/m/mlwagman/AmSC/hmc_runs"
    }
  },
  "hadrons": {
    "Perlmutter": {
      "bin": "UNUSED_FOR_HMC",
      "env": "UNUSED_FOR_HMC"
    }
  }
}
JSON
mkdir -p /pscratch/sd/m/mlwagman/AmSC/hmc_runs
```

(The `hadrons` block is required by `ManagerConfig`'s schema even though
HMCOptimizer doesn't use it.)

### 6b. Dry-run first

Always preview the batch script before paying for an allocation:

```bash
hmc-submit \
  --binary-prefix "$HMC_DRIVER_PREFIX" \
  --integrator omelyan \
  --lattice 8,8,8,8 --beta 6.0 --m-light 0.05 --m-strange 0.1 \
  --md-steps 8 --traj-length 1.0 --n-trajectories 5 \
  --mpi-geom 1,1,1,1 --nodes 1 --time 600 \
  --account m4982 --queue debug \
  --dry-run
```

(Check `hmc-submit --help` for the actual flag spelling — the dry-run
above is illustrative; the real surface is what `argparse` prints.)

Verify the printed `srun` line points at `hmc_driver_omelyan`, has the
right `--grid` and `--mpi`, and that `--n-trajectories 5` made it through.

### 6c. Real submission

Drop `--dry-run` and confirm with the human before submitting — the cost
ledger in SKILL.md §10 caps you at 3 trials without check-in for a
reason. The first real trial answers two independent questions:

1. **Does the IRI client work end-to-end** (auth, mkdir, batch submit,
   poll, log fetch)?
2. **Does the binary run correctly under multi-rank MPI** on a real
   Perlmutter GPU node?

If question 1 fails, the binary never ran — debug auth, not physics.

### 6d. Pull the log back and feed the proposer

```bash
hmc-status   --run-dir <pscratch path printed by hmc-submit>
hmc-report   # summarizes the MLflow study, prints the next proposal
```

A successful Stage 3 ends with `hmc-report` printing a concrete next
`(integrator, dt, md_steps)` based on a single real trial — the same
behavior the local integration test verified on 4⁴.

## 7. What to do after Stage 3

Hand control back to the human with:
- The MLflow run ID of the trial (so they can open it in the UI).
- The proposal `hmc-report` printed.
- Any non-trivial deviations from this doc you had to make (module set,
  paths, etc.) — these are signal that the doc needs an edit, not just a
  one-off workaround.

## 8. Troubleshooting cheat sheet

| Symptom                                              | First thing to check                                |
|------------------------------------------------------|-----------------------------------------------------|
| `pip install -e ../HadronsJobBuilder` fails          | Sibling path: `ls ../HadronsJobBuilder/pyproject.toml` |
| Driver build: `Grid/Grid.h: No such file`            | Wrong `GRID_CONFIG`; re-run step 0                  |
| Driver runtime: `ConstEE() == 1` assertion           | You're using an EO action with WilsonClover — wrong driver source. The shipped driver uses non-EO; if you edited it, undo. |
| CG `did NOT converge` / `nan`                        | Cold-start + small mass + small β → switch to `--starting-type HotStart`, raise `m_l`, or loosen `--rat-lo` / `--rat-hi` |
| IRI `401 Unauthorized` mid-stream                    | Token expired. `_iri.py` should refresh on retry; if not, delete `iriapi_key_path` and redo device-code flow |
| `hmc-submit` exits "cumulative cap exceeded"         | This is intentional. Pass `--allow-cap-override` only after confirming with the human |

---

*This document lives in `HMCOptimizer/docs/perlmutter-bootstrap.md`. Edit
it whenever the bootstrap reality drifts; that's cheaper than the next
agent re-deriving it.*
