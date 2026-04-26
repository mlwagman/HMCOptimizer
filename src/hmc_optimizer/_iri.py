"""Narrow alias over HadronsJobBuilder's workflow_manager.

The only file in HMCOptimizer allowed to import from `femtomeas`. Every other
module in the package imports from here instead. When the underlying IRI client
moves (to `lqcd_workflow/` or standalone), this file is the single point of fix.

Exposed functions use snake_case names matching the rest of HMCOptimizer.
"""
from __future__ import annotations

import functools
from typing import Callable, TypeVar

from femtomeas.workflow_manager.api_general import (
    setupWorkflowAgent,
    remoteMkdir,
    uploadBytes,
    executeBatchJobCompat,
    getJobState,
    cancelJob,
    queryMachineStatus,
    globusTransferStatus,
    globusCopyToMachine,
    globusCopyFromMachine,
    getUserAccountProjects,
    getKnownMachines,
    getMachineQueues,
)
from femtomeas.workflow_manager import globals as _fm_globals

setup_workflow_agent = setupWorkflowAgent
remote_mkdir = remoteMkdir
upload_bytes = uploadBytes
submit_batch_job = executeBatchJobCompat
get_job_state = getJobState
cancel_job = cancelJob
query_machine_status = queryMachineStatus
globus_transfer_status = globusTransferStatus
globus_copy_to_machine = globusCopyToMachine
globus_copy_from_machine = globusCopyFromMachine
get_user_accounts = getUserAccountProjects
get_known_machines = getKnownMachines
get_machine_queues = getMachineQueues


def sandbox_dir(machine: str) -> str:
    """Return the configured sandbox base directory for `machine`.

    Fails loudly if `setup_workflow_agent` has not been called yet.
    """
    if _fm_globals.remote_workdir is None:
        raise RuntimeError(
            "Workflow agent is not configured. "
            "Call `setup_workflow_agent(...)` (or `readManagerConfigFile`) first."
        )
    if machine not in _fm_globals.remote_workdir:
        raise KeyError(f"No sandbox configured for machine {machine!r}")
    return _fm_globals.remote_workdir[machine]


F = TypeVar("F", bound=Callable[..., object])


def with_retry_on_auth(fn: F) -> F:
    """Retry `fn` once on an auth error.

    IRI and MLflow both surface 401s when tokens expire mid-study. The refresh
    logic lives inside `setupWorkflowAgent` / the MLflow client; here we just
    give those one retry before bubbling the exception.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # IRI client doesn't yet raise typed auth errors
            msg = str(e).lower()
            if "401" in msg or "unauthorized" in msg or "expired" in msg:
                return fn(*args, **kwargs)
            raise

    return wrapper  # type: ignore[return-value]
