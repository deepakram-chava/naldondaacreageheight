import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.app.services.nalgonda_processing_engine import engine

print("Starting Rabi 2023...")
engine.process_dataset(2023, "Rabi", "init-job-2023-rabi")
print("Starting Rabi 2024...")
engine.process_dataset(2024, "Rabi", "init-job-2024-rabi")
print("Done seeding Rabi data.")
