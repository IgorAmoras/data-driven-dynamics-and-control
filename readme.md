# Data-Driven Dynamics and Control

Research code for dynamics, control, system identification, and Gaussian Processes.

## Python environment (Windows)

The notebooks should use the repository-local `.venv` created with Python 3.12. Do not install project dependencies into a global or user Python installation.

### Prepare from scratch

Install Python 3.12 and make sure the Python launcher (`py`) can find it. From the repository root, run:

```powershell
pwsh -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

The script creates `.venv` if it does not exist, installs the exact Windows/Python 3.12 versions in `requirements-lock.txt`, and registers the Jupyter kernel `Python (data-driven-dynamics-and-control)`. It checks an existing `.venv` before using it and never enables system site packages.

To activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or invoke it directly without activation:

```powershell
.\.venv\Scripts\python.exe
```

### VS Code and notebooks

Open this repository as the VS Code workspace. The workspace setting prefers `.venv\Scripts\python.exe` for Python. For a notebook, use the kernel picker in the upper right and select **Python (data-driven-dynamics-and-control)**. If it is not listed, run `scripts/setup.ps1` again from the repository root, then reload the VS Code window.

### Adding or changing dependencies

Edit `requirements.txt` to record each direct dependency; keep the scientific stack versions explicit and compatible with Python 3.12. Then activate `.venv` and install only through its interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install <package>
.\.venv\Scripts\python.exe -m pip check
```

After resolving and reviewing the change, update the complete resolved lock from this environment:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding ascii requirements-lock.txt
```

Never generate the lock with a global Python. Commit the direct dependency change and updated `requirements-lock.txt` together so other researchers can reproduce the environment.

### Rebuild after a conflict

First inspect the current `.venv` and preserve it if it contains anything useful. Rename it (for example, to `.venv-backup`) rather than deleting it, then run `scripts/setup.ps1`. The script creates a clean isolated environment and reinstalls the locked versions. `.venv` and common notebook temporary files are excluded by `.gitignore`.
