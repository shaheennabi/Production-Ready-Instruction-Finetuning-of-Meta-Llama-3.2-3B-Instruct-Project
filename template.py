import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# List of files and directories relative to the current root
list_of_files = [
    ".github/workflows/main.yml",
    ".github/FUNDING.yml",
    "notebooks/exploratory/.gitkeep",
    "notebooks/model_training/.gitkeep",
    "notebooks/evaluation/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_processing.py",
    "src/components/model_training.py",
    "src/components/model_evaluation.py",
    "src/configuration/__init__.py",
    "src/constants/__init__.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifacts_entity.py",
    "src/exception/__init__.py",
    "src/exception/exception.py",
    "src/logger/__init__.py",
    "src/logger/logger.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src//utils/__init__.py",
    "src//utils/main_utils.py",
    "results/metrics.py",
    "results/plots.py",
    "scripts/train_model.py",
    "scripts/eval_model.py",
    "config/train_config.py",
    "config/eval_config.py",
    "docs/README.md",  
    "data/.gitkeep",   
    "templates/.gitkeep",  
    ".dockerignore",
    "demo.py",
    "flowcharts/.gitkeep", 
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
]

# File and directory creation logic
for filepath in list_of_files:
    filepath = Path(filepath)

    # Split file directory and filename
    filedir, filename = os.path.split(filepath)

    # Create directories if needed
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    # Create file if it doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filepath} already exists")
