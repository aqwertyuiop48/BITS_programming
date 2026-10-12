# Local Setup

Run all commands from the repository root using Python 3.11. Allow several GB
for packages, data and model artifacts; 16 GB RAM is recommended for full-data
training. Commands below use Windows PowerShell and the explicit virtual
environment interpreter. On other systems use the corresponding `.venv/bin/python`.

## 1. Environment and Tests

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

No activation or execution-policy change is necessary. The test suite uses
synthetic inputs and mocked cloud clients. A passing test suite does not prove
cloud availability or model accuracy. Core requirements do not include the
optional dashboard or managed-monitoring dependencies.

Use 64-bit Python 3.11 even when a newer system Python is installed. Verify with
`.\.venv\Scripts\python.exe --version`; an existing environment keeps the version
used to create it. Select this environment with VS Code's Microsoft Python
extension. For notebooks, also install Microsoft's Jupyter extension and select
the same `.venv` kernel. The dependency list includes notebook support, not
presentation-authoring software.

Plain `pytest` does not enable coverage automatically. To check the configured
coverage threshold, use
`.\.venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing`.
The [recording companion](RECORDING_COMPANION.md) maps each activity to its lab
and records fresh-checkout differences.

## 2. Data Pipeline

Defaults select January 2024 Yellow Taxi records from NYC TLC. Review
[data limitations](DATA_STATEMENT.md) before using the output.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_pipeline --year 2024 --month 1
```

This downloads raw data, cleans trips, creates dense hourly zone counts, builds
features, and writes chronological train/validation/test splits. Outputs are
under `data/raw`, `data/processed`, and `data/features`. The cleaning summary is
`data/processed/pipeline_summary.json`. After a successful download, the same
pipeline can be rerun with `--skip-download` to reuse the cache.

### Exploration Notebook

After the pipeline completes, open
[the exploration notebook](../notebooks/01_explore_dataset.ipynb) in VS Code,
select the project `.venv` kernel, and run cells in order as described in
[L3](L3_DATA_EXPLORATION_LAB.md). It is supplied without execution results or
machine-specific metadata. Running cells calculates in-memory examples; it does
not replace the pipeline's persisted outputs.

Optional browser-based alternative:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks/01_explore_dataset.ipynb
```

The recorded `python -m scripts.build_explore_notebook` preparation command is
included, but overwrites the notebook. Do not run it after adding work you need.

## 3. Training and Registry

Start with one tuning trial per estimator on each problem:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_models --n-trials 1
```

Both XGBoost and LightGBM use the full splits. This is still substantive
training, not a tiny synthetic example. Omitting `--n-trials` uses the configured
25 trials. The winner is selected by validation RMSE; candidate test scores are
logged but must not be used to tune or select models.

The default tracking store is `sqlite:///mlflow.db`, with artifacts under
`mlruns/`. Winners register as `taxi-demand-forecaster` and `taxi-fare-predictor`.
SHAP plots and model cards are generated under `artifacts/`. The optional
`--no-register` flag logs runs but does not create serving registrations.

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The UI at <http://127.0.0.1:5000> does not train or register a model itself.

## 4. API

After preparing both registered models and demand history:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

Use <http://127.0.0.1:8000/docs> for requests. Check readiness in another terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Require JSON `status: ok`, both models loaded, and nonzero history and zones.
The endpoint can return HTTP 200 with degraded JSON. [API examples](API_EXAMPLES.md)
cover fare, demand, recursive forecasting and SHAP requests.

## 5. Batch and Analysis

```powershell
.\.venv\Scripts\python.exe -m scripts.batch_forecast --horizon 3
.\.venv\Scripts\python.exe -m scripts.responsible.run_analysis
.\.venv\Scripts\python.exe -m scripts.monitoring.demo_drift_report --problem all --mode drift
```

Batch forecasts append rows to `predictions.db`. Repeated runs do not replace
earlier output. The `latest` model reference is mutable and is not a resolved
numeric model lineage identifier.

Responsible analysis reads registered models and held-out test splits, writing
`docs/model_cards/` and `artifacts/responsible/`. Drift output is written under
`artifacts/drift/`; `--mode drift` deliberately shifts two numeric features and
is not evidence of a production incident. `--mode clean` uses unshifted samples.
Analysis commands can replace reports in those locations; preserve any evidence
you need before regenerating it. Global SHAP plots do not explain one named trip.

## 6. Dashboard

Generate the analysis outputs above, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
$env:API_BASE_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501>. The dashboard combines live API calls with local
report files; their timestamps and model identities may differ. Missing reports
are displayed as unavailable, not replaced with bundled historical results.
Revenue panels use an illustrative average-fare assumption, not realized revenue.
If a port is occupied, choose a free one and update the API URL consistently.

## 7. Optional Containers

Docker must be installed separately. First complete data preparation and model
registration. Create a portable copy of the registry for Linux artifact paths:

```powershell
.\.venv\Scripts\python.exe -m scripts.make_mlflow_portable --src mlflow.db --dest mlflow.container.db --new-base file:///app/mlruns
docker compose -f docker/docker-compose.yml up --build
```

The Compose services mount the portable database, `mlruns/`, and data. Stop the
local API/MLflow processes first if ports 8000/5000 are in use. Regenerate the
portable database after new local registrations. The dashboard image also
requires generated cards, responsible/drift reports, and the zone lookup; it
is not built by the API/MLflow Compose stack.

## 8. Optional Cloud and Configuration

Central configuration is in `config/config.yaml`. Copy `.env.example` to `.env`
only when you need local overrides and no `.env` already exists. Do not commit
credentials, raw data, models, state files or local databases. The dashboard
reads `API_BASE_URL` from its process environment, so set it in the shell.

See [the infrastructure runbook](../infra/terraform/README.md) before any cloud
operation. `requirements-monitoring.txt` supplies AWS SDK and optional reporting
dependencies; it is not needed for core local tests. A managed job is not the
same as an always-on API. Neither a drift flag nor a passing quality gate grants
automatic deployment permission.

## 9. Combined End-to-End PowerShell Run

From the repository root, paste this entire block into PowerShell to run the
setup, tests, pipeline, training, analysis and serving steps in order. A failed
native command stops the run. JupyterLab, MLflow, the API and Streamlit are
started as background processes so the sequence can continue; their output is
written to a uniquely named folder under `$env:TEMP`. The API health check must
report `status: ok`. On success, the services remain running after the block
finishes. If a command fails, any services already started by the block are
stopped.

This is a full run: it downloads data, trains models, and can take substantial
time and disk space. Open JupyterLab, MLflow, the API, and Streamlit at their
documented local URLs after the block completes.

```powershell
$ErrorActionPreference = 'Stop'
$script:Services = [System.Collections.Generic.List[object]]::new()
$script:Completed = $false
$script:Python = Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
$script:LogDir = Join-Path $env:TEMP ("taxi-forecasting-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $script:LogDir | Out-Null

function Assert-Services {
    foreach ($service in $script:Services) {
        $service.Process.Refresh()
        if ($service.Process.HasExited) {
            $stderr = if (Test-Path $service.Stderr) { Get-Content $service.Stderr -Raw } else { '' }
            throw "$($service.Name) exited with code $($service.Process.ExitCode). $stderr Logs: $script:LogDir"
        }
    }
}

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments)
    Assert-Services
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
    Assert-Services
}

function Start-CheckedService {
    param([string]$Name, [string[]]$Arguments)
    Assert-Services
    $stdout = Join-Path $script:LogDir "$Name.stdout.log"
    $stderr = Join-Path $script:LogDir "$Name.stderr.log"
    $process = Start-Process -FilePath $script:Python -ArgumentList $Arguments `
        -WorkingDirectory (Get-Location).Path -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    $script:Services.Add([pscustomobject]@{
        Name = $Name
        Process = $process
        Stdout = $stdout
        Stderr = $stderr
    }) | Out-Null
    Start-Sleep -Seconds 2
    Assert-Services
    Write-Host "$Name started. Logs: $script:LogDir"
}

try {
    Invoke-Checked 'py' @('-3.11', '--version')
    Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.venv')
    if (-not (Test-Path $script:Python)) { throw "Virtual environment Python was not created: $script:Python" }

    Invoke-Checked $script:Python @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-Checked $script:Python @('-m', 'pip', 'install', '-r', 'requirements.txt')
    Invoke-Checked $script:Python @('-m', 'pytest')
    Invoke-Checked $script:Python @('-m', 'ruff', 'check', '.')
    Invoke-Checked $script:Python @('-m', 'pytest', '--cov=src', '--cov-report=term-missing')
    Invoke-Checked $script:Python @('-m', 'scripts.run_pipeline', '--year', '2024', '--month', '1')

    Start-CheckedService 'JupyterLab' @('-m', 'jupyterlab', 'notebooks/01_explore_dataset.ipynb')
    Invoke-Checked $script:Python @('-m', 'scripts.train_models', '--n-trials', '1')
    Start-CheckedService 'MLflow' @('-m', 'mlflow', 'ui', '--backend-store-uri', 'sqlite:///mlflow.db', '--host', '127.0.0.1', '--port', '5000')
    Start-CheckedService 'API' @('-m', 'uvicorn', 'src.serving.api:app', '--host', '127.0.0.1', '--port', '8000')

    $health = $null
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Assert-Services
        try {
            $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 5
            break
        } catch {
            if ($attempt -eq 29) { throw }
            Start-Sleep -Seconds 2
        }
    }
    if ($health.status -ne 'ok') { throw "API health check did not report status 'ok'." }

    Invoke-Checked $script:Python @('-m', 'scripts.batch_forecast', '--horizon', '3')
    Invoke-Checked $script:Python @('-m', 'scripts.responsible.run_analysis')
    Invoke-Checked $script:Python @('-m', 'scripts.monitoring.demo_drift_report', '--problem', 'all', '--mode', 'drift')
    Invoke-Checked $script:Python @('-m', 'pip', 'install', '-r', 'requirements-dashboard.txt')

    $env:API_BASE_URL = 'http://127.0.0.1:8000'
    Start-CheckedService 'Streamlit' @('-m', 'streamlit', 'run', 'dashboard/app.py', '--server.address', '127.0.0.1', '--server.port', '8501')
    $script:Completed = $true
} finally {
    if (-not $script:Completed) {
        foreach ($service in $script:Services) {
            $service.Process.Refresh()
            if (-not $service.Process.HasExited) {
                Stop-Process -Id $service.Process.Id -Force
            }
        }
    }
}
```