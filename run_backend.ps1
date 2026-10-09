# run_backend.ps1
$env:ENVIRONMENT="development"
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --reload
