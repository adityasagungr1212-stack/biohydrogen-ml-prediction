import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
scripts = [
    "models/xgboost_model.py",
    "models/random_forest_model.py",
    "models/svr_model.py",
    "models/knn_model.py",
    "models/lightgbm_model.py",
    "models/catboost_model.py",
]

for script in scripts:
    print(f"\n===== Running {script} =====")
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Model failed: {script}")

print("\nAll models completed successfully.")
