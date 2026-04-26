# Auth quick reference — IRI, MLflow, Globus

The skill treats auth as someone else's problem and fails fast on 401s
with a clear remediation path. This file is that path.

## IRI API (DOE facility API)

The IRI key file is pointed at via the manager config:

```json
{
  "workflow": {
    "sfapi_key_path": "/path/to/sfapi_key",
    "iriapi_key_path": "/path/to/iriapi_key",
    "sandbox_directories": { "Perlmutter": "/pscratch/sd/.../sandbox" }
  },
  ...
}
```

The IRI key is created on first use via the Globus OAuth2 device-code
flow (`iri_api.py:setupIRIapi` → `interactive_login`) — the first
invocation prints an auth URL and a code to paste. Subsequent invocations
load the cached tokens from `iriapi_key_path` and refresh via
`oauth2_refresh_token` (`iri_api.py:refresh_tokens`). The only scope
requested up-front is the IRI API scope
(`iri_api.py:REQUIRED_SCOPES`); Globus will prompt separately for the
transfer scope on first transfer.

**401 recovery.** `hmc_optimizer._iri.with_retry_on_auth` wraps every
call and retries once. If that fails:

1. Delete `iriapi_key_path` and rerun — the device-code flow re-runs.
2. Check that your Globus account has an active AmSC / Perlmutter
   identity linkage (usually a one-time setup via MyAmSC).
3. Confirm the IRI service hasn't rotated its OAuth client —
   `setupIRIapi` should surface an `invalid_client` error if so.

## MLflow

The demo uses a local MLflow server (`mlflow server --port 5000`).
Point at it with:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

For a remote / gateway-hosted tracking server, set the same env var to
its URL. MLflow's Python client picks up credentials via its standard
env vars and config file (`MLFLOW_TRACKING_USERNAME`,
`MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_TOKEN` are all consumed
by the HTTP client per MLflow's docs); this package does not wrap or
override that behavior.

If the tracking server is unreachable, `track.log_trial` raises a
requests-level error. It is **not** retried automatically — the cost
of a silent retry loop here (hidden token expiry during a long study)
outweighs the convenience.

## Globus transfers

`iri_api.py` keeps a small hard-coded map of machine name → Globus
endpoint name at `iri_api.py:machine_globus_endpoints` (currently
`{"Perlmutter": "perlmutter"}`). There is no auto-resolution from IRI
facility metadata — if you need a new machine, add it to that map.

The first transfer attempt will trigger a Globus consent prompt for
the transfer scope (not requested up-front at login time). If a
transfer 403s repeatedly, rerun `globus login --force` and retry.

## Resume after auth failure

`scripts/hmc_submit.py` writes a manifest JSON on successful submit.
On an unrecoverable auth failure mid-study, the recommended pattern is:

1. Kill the running driver.
2. Fix auth (steps above).
3. Re-run `hmc_submit.py` — it re-reads MLflow history, and the
   proposer picks up where it left off with no state lost.

A dedicated `~/.hmc_optimizer/resume/<session_id>.json` file is not
yet implemented; MLflow + the manifest JSON is the durable state.
