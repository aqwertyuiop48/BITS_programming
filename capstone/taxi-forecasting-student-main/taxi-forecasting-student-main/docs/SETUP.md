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

GitHub Actions builds the API Dockerfile's `ci` target and checks that the
application imports in the image. Its full end-to-end job then downloads the
data, trains/registers models, runs batch inference, generates Responsible-AI
and synthetic drift reports, builds and runs the production API image, makes
live inference requests, and checks the dashboard responds. This can take
hours. Logs and generated reports are uploaded as the `capstone-e2e-evidence`
workflow artifact. The full job uses a temporary GitHub runner and does not
publish an image or deploy a persistent service; the AWS CodeBuild deployment
guide is separate. Do not commit generated model/data artifacts or credentials
to make CI pass.

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
setup, tests, pipeline, training, analysis and serving steps in order. It keeps
an existing `.venv` (as on the preconfigured AWS course image) and creates one
only if missing. A failed native command stops the run. JupyterLab, MLflow, the
API, and Streamlit are started as background processes so the sequence can
continue; when Docker is available and selected, the API and MLflow UI run in
Compose instead. Output and API inference evidence are written to a uniquely
named folder under `$env:TEMP`. The API health check and live fare/demand
requests must succeed. The block also verifies Responsible-AI and drift outputs
and copies them into the evidence folder. On success, services remain running
after the block finishes. If a command fails, services started by the block are
stopped.

At the first prompt, answer `Y` only on a supported Docker host with Docker
Engine/Desktop running and Compose v2 installed. This builds and starts the
full API image using the generated models, then sends the API smoke requests to
the container. The preconfigured Windows Server course image does not support
Docker Desktop; on that machine answer `N` for the local run, and run the Docker
option on a supported Docker host. This script does not publish the image to a
registry or create a remote deployment.

At the end, you can optionally publish the source to a new public GitHub
repository. If GitHub CLI (`gh`) is installed and authenticated, the block can
create the repository; otherwise, create an empty public repository in GitHub
first and provide its HTTPS URL when prompted. Review the staged files before
confirming publication; never publish credentials or personal data. The Docker
option runs containers locally; it does not publish an image to a registry,
deploy to a remote host, or prove a hosted CI/CD run.

The AWS image folder is a source snapshot, not a Git checkout. When you choose
publication, this block runs `git init -b main`, stages and displays the files,
commits only after you confirm, then creates or connects the public GitHub
repository and pushes the current branch. Before running it, make sure Git is
installed and you can authenticate to GitHub from the remote desktop. Copy the
capstone `.github/workflows/ci.yml` and the updated `docker/Dockerfile` containing
the `ci` target into this source folder before publishing; the block checks for
both. With no `gh` CLI, create an empty public repository in your browser first;
do not add a README or license there, then enter its HTTPS URL at the prompt.

The commit prompts for a display name and email only if Git has no identity
configured for this repository. A GitHub `noreply` address can keep your
personal email out of the public commit.

This is a full run: it downloads data, trains models, and can take substantial
time and disk space. Open JupyterLab, MLflow, the API, and Streamlit at their
documented local URLs after the block completes. With Docker selected, API and
MLflow are the Compose containers; Streamlit and JupyterLab remain local.

```powershell
$ErrorActionPreference = 'Stop'
$script:Services = [System.Collections.Generic.List[object]]::new()
$script:Completed = $false
$script:UseDocker = $false
$script:DockerStarted = $false
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
    $dockerChoice = Read-Host 'Run the API and MLflow UI in Docker Compose on this supported Docker host? [y/N]'
    $script:UseDocker = $dockerChoice -match '^(?i:y|yes)$'
    if ($script:UseDocker) {
        Invoke-Checked 'docker' @('info', '--format', '{{.ServerVersion}}')
        Invoke-Checked 'docker' @('compose', 'version')
    }

    if (-not (Test-Path $script:Python)) {
        Invoke-Checked 'py' @('-3.11', '--version')
        Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.venv')
        if (-not (Test-Path $script:Python)) { throw "Virtual environment Python was not created: $script:Python" }
        Invoke-Checked $script:Python @('-m', 'pip', 'install', '-r', 'requirements.txt')
    } else {
        Invoke-Checked $script:Python @('-m', 'pip', 'check')
    }

    Invoke-Checked $script:Python @('-m', 'pytest')
    Invoke-Checked $script:Python @('-m', 'ruff', 'check', '.')
    Invoke-Checked $script:Python @('-m', 'pytest', '--cov=src', '--cov-report=term-missing')
    Invoke-Checked $script:Python @('-m', 'scripts.run_pipeline', '--year', '2024', '--month', '1')
    if (-not (Test-Path 'data\processed\pipeline_summary.json')) {
        throw 'Pipeline summary was not generated.'
    }
    Copy-Item 'data\processed\pipeline_summary.json' (Join-Path $script:LogDir 'pipeline_summary.json')

    Start-CheckedService 'JupyterLab' @('-m', 'jupyterlab', 'notebooks/01_explore_dataset.ipynb')
    Invoke-Checked $script:Python @('-m', 'scripts.train_models', '--n-trials', '1')
    if (-not (Test-Path 'mlflow.db') -or -not (Test-Path 'mlruns')) {
        throw 'Training did not produce the expected MLflow registry files.'
    }
    if ($script:UseDocker) {
        Invoke-Checked $script:Python @(
            'scripts/make_mlflow_portable.py',
            '--src', 'mlflow.db',
            '--dest', 'mlflow.container.db',
            '--new-base', 'file:///app/mlruns'
        )
        if (-not (Test-Path 'mlflow.container.db') -or -not (Test-Path 'data\processed\demand_hourly.parquet')) {
            throw 'Docker image inputs are missing: expected mlflow.container.db and demand history.'
        }
        $script:DockerStarted = $true
        Invoke-Checked 'docker' @('compose', '-f', 'docker/docker-compose.yml', 'up', '--build', '-d')
        Invoke-Checked 'docker' @('compose', '-f', 'docker/docker-compose.yml', 'ps')
    } else {
        Start-CheckedService 'MLflow' @('-m', 'mlflow', 'ui', '--backend-store-uri', 'sqlite:///mlflow.db', '--host', '127.0.0.1', '--port', '5000')
        Start-CheckedService 'API' @('-m', 'uvicorn', 'src.serving.api:app', '--host', '127.0.0.1', '--port', '8000')
    }

    $health = $null
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Assert-Services
        try {
            $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 5
            if ($health.status -eq 'ok') { break }
            Write-Host 'API is responding but models are not ready yet.'
            $health = $null
        } catch {
            if ($attempt -eq 29) { throw }
        }
        if ($attempt -lt 29) { Start-Sleep -Seconds 2 }
    }
    if ($null -eq $health -or $health.status -ne 'ok') { throw "API health check did not report status 'ok'." }
    $health | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $script:LogDir 'api-health.json')
    Write-Host 'API health:'
    $health | ConvertTo-Json -Depth 8

    $fare = @{
        trip_distance = 17.0
        pu_location_id = 132
        do_location_id = 230
        passenger_count = 1
        pickup_datetime = '2024-01-15T18:30:00'
    } | ConvertTo-Json
    $fareResult = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict/fare' `
        -ContentType 'application/json' -Body $fare -TimeoutSec 60
    Write-Host 'Live fare prediction:'
    $fareResult | ConvertTo-Json -Depth 8

    $demand = @{ zone = 132; target_hour = '2024-02-01T00:00:00' } | ConvertTo-Json
    $demandResult = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict/demand' `
        -ContentType 'application/json' -Body $demand -TimeoutSec 60
    Write-Host 'Live demand prediction:'
    $demandResult | ConvertTo-Json -Depth 8

    [ordered]@{
        api_health = $health
        fare_prediction = $fareResult
        demand_prediction = $demandResult
    } | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $script:LogDir 'inference-evidence.json')

    $batchSummary = Invoke-Checked $script:Python @('-m', 'scripts.batch_forecast', '--horizon', '3')
    $batchSummary | Set-Content (Join-Path $script:LogDir 'batch-forecast.json')

    Invoke-Checked $script:Python @('-m', 'scripts.responsible.run_analysis')
    $responsibleFiles = @(
        'docs\model_cards\demand.md',
        'docs\model_cards\fare.md',
        'artifacts\responsible\demand_fairness.json',
        'artifacts\responsible\fare_fairness.json'
    )
    foreach ($file in $responsibleFiles) {
        if (-not (Test-Path $file)) { throw "Responsible-AI output was not generated: $file" }
    }
    $responsibleEvidence = Join-Path $script:LogDir 'responsible-ai'
    New-Item -ItemType Directory -Path $responsibleEvidence | Out-Null
    foreach ($file in $responsibleFiles) {
        Copy-Item $file (Join-Path $responsibleEvidence (Split-Path $file -Leaf))
    }
    Write-Host "Responsible-AI cards and fairness reports: $responsibleEvidence"

    Invoke-Checked $script:Python @('-m', 'scripts.monitoring.demo_drift_report', '--problem', 'all', '--mode', 'drift')
    $driftFiles = @(
        'artifacts\drift\demand_drift_metrics.json',
        'artifacts\drift\fare_drift_metrics.json'
    )
    foreach ($file in $driftFiles) {
        if (-not (Test-Path $file)) { throw "Drift report was not generated: $file" }
    }
    $driftEvidence = Join-Path $script:LogDir 'drift'
    New-Item -ItemType Directory -Path $driftEvidence | Out-Null
    foreach ($file in $driftFiles) {
        Copy-Item $file (Join-Path $driftEvidence (Split-Path $file -Leaf))
    }
    Write-Host "Synthetic drift demo reports: $driftEvidence"

    Invoke-Checked $script:Python @('-m', 'pip', 'install', '-r', 'requirements-dashboard.txt')

    $env:API_BASE_URL = 'http://127.0.0.1:8000'
    Start-CheckedService 'Streamlit' @('-m', 'streamlit', 'run', 'dashboard/app.py', '--server.address', '127.0.0.1', '--server.port', '8501')
    $dashboardReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Assert-Services
        try {
            $dashboard = Invoke-WebRequest -Uri 'http://127.0.0.1:8501/' -UseBasicParsing -TimeoutSec 5
            if ($dashboard.StatusCode -eq 200) {
                $dashboardReady = $true
                break
            }
        } catch {
            if ($attempt -eq 29) { throw }
            Start-Sleep -Seconds 2
        }
    }
    if (-not $dashboardReady) { throw 'Dashboard did not become ready at http://127.0.0.1:8501/.' }
    Write-Host "Run evidence: $script:LogDir"
    Write-Host 'Open the API at http://127.0.0.1:8000/docs and dashboard at http://127.0.0.1:8501/'
    $script:Completed = $true
} finally {
    if (-not $script:Completed) {
        foreach ($service in $script:Services) {
            $service.Process.Refresh()
            if (-not $service.Process.HasExited) {
                Stop-Process -Id $service.Process.Id -Force
            }
        }
        if ($script:DockerStarted) {
            & docker compose -f docker/docker-compose.yml down
            if ($LASTEXITCODE -ne 0) {
                Write-Warning 'Docker Compose cleanup failed. Review the Docker error and stop the project containers before retrying.'
            }
        }
    }
}

if ($script:Completed) {
    $publish = Read-Host 'To publish source to a new PUBLIC GitHub repository, type PUBLIC; press Enter to skip'
    if ($publish -ceq 'PUBLIC') {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            throw 'Git is not installed or not on PATH. Install Git for Windows, reopen PowerShell, and rerun the publication steps.'
        }
        if (-not (Test-Path 'README.md') -or -not (Test-Path '.gitignore')) {
            throw 'Run this block from the capstone project root containing README.md and .gitignore.'
        }
        if (-not (Test-Path '.github\workflows\ci.yml')) {
            throw 'The capstone CI workflow is missing. Copy .github\workflows\ci.yml into this project folder before publishing.'
        }
        if (-not (Select-String -Path 'docker\Dockerfile' -Pattern '^FROM runtime-base AS ci$' -Quiet)) {
            throw 'The Docker CI target is missing. Copy the updated docker\Dockerfile into this project folder before publishing.'
        }

        $useGh = $false
        if (Get-Command gh -ErrorAction SilentlyContinue) {
            & gh auth status
            $useGh = ($LASTEXITCODE -eq 0)
            if (-not $useGh) {
                Write-Warning 'GitHub CLI is not authenticated. Create an empty public repository in GitHub and use its HTTPS URL when prompted.'
            }
        }
        if (-not (Test-Path '.git')) {
            Invoke-Checked 'git' @('init', '-b', 'main')
        }
        $gitName = (& git config user.name 2>$null)
        if ([string]::IsNullOrWhiteSpace($gitName)) {
            $gitName = Read-Host 'Enter the name to display on your Git commit'
            if ([string]::IsNullOrWhiteSpace($gitName)) { throw 'A Git commit name is required.' }
            Invoke-Checked 'git' @('config', 'user.name', $gitName)
        }
        $gitEmail = (& git config user.email 2>$null)
        if ([string]::IsNullOrWhiteSpace($gitEmail)) {
            $gitEmail = Read-Host 'Enter a commit email (use your GitHub noreply address if you want to keep your email private)'
            if ([string]::IsNullOrWhiteSpace($gitEmail)) { throw 'A Git commit email is required.' }
            Invoke-Checked 'git' @('config', 'user.email', $gitEmail)
        }
        $branch = (& git branch --show-current).Trim()
        if ($LASTEXITCODE -ne 0) { throw 'Could not read the current Git branch.' }
        if ([string]::IsNullOrWhiteSpace($branch)) {
            Invoke-Checked 'git' @('switch', '-c', 'main')
            $branch = 'main'
        }
        Invoke-Checked 'git' @('add', '--all')
        Write-Host 'Review this staged-file list. Do not publish secrets, credentials, or personal data:'
        Invoke-Checked 'git' @('status', '--short')
        Invoke-Checked 'git' @('diff', '--cached', '--check')
        $confirmCommit = Read-Host 'If the staged files are safe to publish, type COMMIT'
        if ($confirmCommit -ceq 'COMMIT') {
            Invoke-Checked 'git' @('commit', '-m', 'Submit completed taxi forecasting capstone')
            if ($useGh) {
                $repoName = Read-Host 'Enter a new GitHub repository name (for example taxi-forecasting-capstone)'
                if ([string]::IsNullOrWhiteSpace($repoName) -or $repoName -notmatch '^[A-Za-z0-9._-]+$') {
                    throw 'Repository name is empty or contains unsupported characters.'
                }
                $githubUser = (& gh api user --jq .login).Trim()
                if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($githubUser)) {
                    throw 'Could not determine the authenticated GitHub username.'
                }
                Invoke-Checked 'gh' @('repo', 'create', "$githubUser/$repoName", '--public', '--source', '.', '--remote', 'origin')
                Invoke-Checked 'git' @('push', '-u', 'origin', $branch)
            } else {
                $repoUrl = Read-Host 'Enter the HTTPS URL of your empty public GitHub repository'
                if ($repoUrl -notmatch '^https://github\.com/[^/]+/[^/]+(?:\.git)?/?$') {
                    throw 'Enter a repository URL in the form https://github.com/username/repository.'
                }
                $existingOrigin = (& git remote get-url origin 2>$null)
                if ($LASTEXITCODE -eq 0 -and $existingOrigin) {
                    throw "Git remote 'origin' already exists ($existingOrigin). Verify it before publishing; refusing to overwrite it."
                }
                Invoke-Checked 'git' @('remote', 'add', 'origin', $repoUrl)
                Invoke-Checked 'git' @('push', '-u', 'origin', $branch)
            }
            Invoke-Checked 'git' @('remote', '-v')
            Invoke-Checked 'git' @('rev-parse', '--short', 'HEAD')
        } else {
            Write-Host 'GitHub publication skipped; no commit or push was made.'
        }
    }
}
```