"""Proposers for HMC hyperparameter sweeps.

Two strategies live side-by-side behind a shared `Proposer` ABC:

- `RuleProposer` (default) — deterministic heuristics distilled from the TXQCD
  tuning sessions. Small state, transparent reasoning, cheap to run locally.
- `OptunaProposer` (`--strategy optuna`) — TPE over `(dt, md_steps)` at a fixed
  integrator, warm-started from any existing MLflow history for the experiment.

Neither touches MLflow or IRI directly; callers read history from MLflow, pass
it as a `Sequence[TrialResult]`, and submit the returned `HMCParams` via
`submit.py`.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

Integrator = Literal["leapfrog", "force_gradient", "omelyan"]

_COST_MULTIPLIER: dict[Integrator, float] = {
    "leapfrog": 1.0,
    "omelyan": 1.0,
    "force_gradient": 1.33,
}

_ACCEPT_LOW = 0.70
_ACCEPT_HIGH = 0.90
_DT_FLOOR = 0.005
_MD_STEPS_CEIL = 100


class IntractableError(RuntimeError):
    """Raised when the search space has been exhausted without convergence."""


@dataclass(frozen=True)
class HMCParams:
    integrator: Integrator
    dt: float
    md_steps: int
    traj_length: float

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.md_steps <= 0 or self.traj_length <= 0:
            raise ValueError(f"Non-positive HMCParams field: {self}")


@dataclass(frozen=True)
class TrialResult:
    params: HMCParams
    accept: float
    dH: float
    plaq: float
    wall_per_traj_s: float
    wall_per_accept_s: float
    n_trajectories: int


def _in_accept_band(accept: float) -> bool:
    return _ACCEPT_LOW <= accept <= _ACCEPT_HIGH


def _clamp_dt(dt: float) -> float:
    if dt < _DT_FLOOR:
        raise IntractableError(
            f"Proposed dt={dt:g} below floor {_DT_FLOOR}. "
            "Pre-conditioner / Hasenbusch ladder likely needs revisiting."
        )
    return dt


def _md_steps_for(dt: float, traj_length: float) -> int:
    steps = max(1, round(traj_length / dt))
    if steps > _MD_STEPS_CEIL:
        raise IntractableError(
            f"md_steps={steps} exceeds ceiling {_MD_STEPS_CEIL}."
        )
    return steps


class Proposer(ABC):
    strategy_name: str = "abstract"

    @abstractmethod
    def propose(self, history: Sequence[TrialResult]) -> HMCParams | None:
        """Return the next `HMCParams` to try, or `None` if converged."""

    def mlflow_tags(self) -> dict[str, str]:
        return {"strategy": self.strategy_name}


class RuleProposer(Proposer):
    """Deterministic, explainable proposer.

    Cold start reuses the TXQCD `Mobius2p1f_EOFA_96I_hmc.cc` starting
    values (integrator `MinimumNorm2`, MDsteps=12, trajL=0.5) — not
    claimed optimal for any other physics point, just a non-arbitrary
    first trial. Subsequent steps are single-trial reactions
    (halve/grow dt) until two same-integrator trials exist, at which
    point we secant-step on `accept(log dt)`.
    """

    strategy_name = "rule"

    def __init__(
        self,
        *,
        allow_force_gradient: bool = False,
        target_traj_length: float = 0.5,
        cold_md_steps: int = 12,
    ) -> None:
        self.allow_force_gradient = allow_force_gradient
        self.target_traj_length = target_traj_length
        self.cold_md_steps = cold_md_steps

    def propose(self, history: Sequence[TrialResult]) -> HMCParams | None:
        if not history:
            return HMCParams(
                integrator="omelyan",
                dt=self.target_traj_length / self.cold_md_steps,
                md_steps=self.cold_md_steps,
                traj_length=self.target_traj_length,
            )

        if self._converged(history):
            return None

        last = history[-1]
        same = [h for h in history if h.params.integrator == last.params.integrator]

        if len(same) >= 2:
            candidate = self._secant_step(same)
            if candidate is not None:
                return candidate

        return self._single_trial_reaction(last, history)

    def _single_trial_reaction(
        self, last: TrialResult, history: Sequence[TrialResult]
    ) -> HMCParams:
        params = last.params
        if last.accept < 0.6:
            new_dt = _clamp_dt(params.dt * 0.5)
        elif last.accept > 0.95:
            new_dt = params.dt * 1.3
        elif _in_accept_band(last.accept) and last.dH > 1.0:
            # Acceptance is fine but integrator is noisy — longer integration
            # won't change dH much, so hold dt and escalate integrator instead.
            return self._maybe_escalate(last, history)
        else:
            new_dt = params.dt * 0.85

        return HMCParams(
            integrator=params.integrator,
            dt=new_dt,
            md_steps=_md_steps_for(new_dt, params.traj_length),
            traj_length=params.traj_length,
        )

    def _maybe_escalate(
        self, last: TrialResult, history: Sequence[TrialResult]
    ) -> HMCParams:
        leapfrog_fails = sum(
            1 for h in history
            if h.params.integrator == "leapfrog" and not _in_accept_band(h.accept)
        )
        if last.params.integrator == "leapfrog" and leapfrog_fails >= 3:
            return HMCParams(
                integrator="omelyan",
                dt=last.params.dt,
                md_steps=last.params.md_steps,
                traj_length=last.params.traj_length,
            )
        if (
            last.params.integrator == "omelyan"
            and self.allow_force_gradient
            and last.dH > 1.5
        ):
            return HMCParams(
                integrator="force_gradient",
                dt=last.params.dt * 1.2,
                md_steps=_md_steps_for(last.params.dt * 1.2, last.params.traj_length),
                traj_length=last.params.traj_length,
            )
        # No escalation available — just tighten dt.
        new_dt = _clamp_dt(last.params.dt * 0.85)
        return HMCParams(
            integrator=last.params.integrator,
            dt=new_dt,
            md_steps=_md_steps_for(new_dt, last.params.traj_length),
            traj_length=last.params.traj_length,
        )

    def _secant_step(self, trials: Sequence[TrialResult]) -> HMCParams | None:
        a, b = trials[-2], trials[-1]
        if a.params.dt == b.params.dt:
            return None
        target = 0.80
        x1, y1 = math.log(a.params.dt), a.accept
        x2, y2 = math.log(b.params.dt), b.accept
        if y2 == y1:
            return None
        x_new = x2 + (target - y2) * (x2 - x1) / (y2 - y1)
        dt_new = math.exp(x_new)
        dt_new = min(dt_new, b.params.dt * 1.5)
        dt_new = max(dt_new, b.params.dt * 0.5)
        dt_new = _clamp_dt(dt_new)
        return HMCParams(
            integrator=b.params.integrator,
            dt=dt_new,
            md_steps=_md_steps_for(dt_new, b.params.traj_length),
            traj_length=b.params.traj_length,
        )

    @staticmethod
    def _converged(history: Sequence[TrialResult]) -> bool:
        good = [h for h in history if _in_accept_band(h.accept) and h.dH < 1.0]
        if len(good) < 2:
            return False
        last_two = good[-2:]
        wpa = [h.wall_per_accept_s for h in last_two]
        if wpa[0] == 0:
            return False
        return abs(wpa[1] - wpa[0]) / wpa[0] < 0.10


class OptunaProposer(Proposer):
    """TPE over `(dt, md_steps)` at a fixed integrator.

    The integrator is chosen up-front by a short bakeoff (see SKILL.md §10);
    Optuna then refines the continuous knobs. Prior rule-based history for the
    same experiment is replayed into the study to warm-start TPE.
    """

    strategy_name = "optuna"

    def __init__(
        self,
        *,
        integrator: Integrator = "omelyan",
        storage: str | None = None,
        study_name: str = "hmc_tune",
        n_trials_budget: int = 20,
        traj_length: float = 0.48,
    ) -> None:
        self.integrator = integrator
        self.storage = storage
        self.study_name = study_name
        self.n_trials_budget = n_trials_budget
        self.traj_length = traj_length
        self._study = None
        self._last_ask = None

    def _ensure_study(self, history: Sequence[TrialResult]):
        if self._study is not None:
            return self._study
        try:
            import optuna
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "OptunaProposer requires `optuna`. "
                "Install via `pip install 'hmc-optimizer[optuna]'`."
            ) from e

        sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=0)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,
        )

        if not study.trials and history:
            for h in history:
                if h.params.integrator != self.integrator:
                    continue
                trial = optuna.trial.create_trial(
                    params={"dt": h.params.dt, "md_steps": h.params.md_steps},
                    distributions={
                        "dt": optuna.distributions.FloatDistribution(
                            _DT_FLOOR, 0.2, log=True
                        ),
                        "md_steps": optuna.distributions.IntDistribution(4, 48),
                    },
                    value=self._objective(h),
                    user_attrs={"dH": h.dH, "plaq": h.plaq, "replayed": True},
                )
                study.add_trial(trial)

        self._study = study
        return study

    @staticmethod
    def _objective(r: TrialResult) -> float:
        penalty = (
            1e6 * max(0.0, _ACCEPT_LOW - r.accept)
            + 1e6 * max(0.0, r.accept - _ACCEPT_HIGH)
        )
        return r.wall_per_accept_s + penalty

    def propose(self, history: Sequence[TrialResult]) -> HMCParams | None:
        study = self._ensure_study(history)
        if len(study.trials) >= self.n_trials_budget and self._best_satisfied(study):
            return None

        trial = study.ask(
            {
                "dt": __import__("optuna").distributions.FloatDistribution(
                    _DT_FLOOR, 0.2, log=True
                ),
                "md_steps": __import__("optuna").distributions.IntDistribution(
                    4, 48
                ),
            }
        )
        self._last_ask = trial
        return HMCParams(
            integrator=self.integrator,
            dt=float(trial.params["dt"]),
            md_steps=int(trial.params["md_steps"]),
            traj_length=self.traj_length,
        )

    def tell(self, result: TrialResult) -> None:
        if self._study is None or self._last_ask is None:
            raise RuntimeError("OptunaProposer.tell called before propose")
        self._study.tell(self._last_ask, self._objective(result))
        self._last_ask = None

    def _best_satisfied(self, study) -> bool:
        try:
            best = study.best_trial
        except ValueError:
            return False
        last5 = study.trials[-5:]
        if len(last5) < 5:
            return False
        improvements = [best.value - t.value for t in last5]
        return max(improvements) < 0.05 * best.value

    def mlflow_tags(self) -> dict[str, str]:
        return {
            "strategy": "optuna",
            "optuna_study": self.study_name,
        }


def make_proposer(
    strategy: Literal["rule", "optuna"] = "rule", **kwargs
) -> Proposer:
    if strategy == "rule":
        return RuleProposer(**kwargs)
    if strategy == "optuna":
        return OptunaProposer(**kwargs)
    raise ValueError(f"Unknown proposer strategy: {strategy!r}")


def hmc_params_as_dict(p: HMCParams) -> dict:
    return asdict(p)
