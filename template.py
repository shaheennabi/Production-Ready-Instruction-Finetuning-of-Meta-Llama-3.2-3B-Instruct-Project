import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# List of files and directories relative to the current root
list_of_files = [
    "PRODUCTION-READY-INSTRUCTION-FINETUNING-OF-META-Llama-3.2-3B Instruct",
    ".github/FUNDING.yml",
    "docs/1. Understanding Instruction Finetuning.md",
    "docs/2. reward_model.md",
    "docs/3. RLHF with PPO.md",
    "docs/4. Direct Preference Optimization.md",
    "docs/5. Understanding ULMA.md",
    "docs/6. Parameter Efficient Finetuning.md",
    "docs/7. Low Rank Adaptation(LORA).md",
    "docs/8. Quantized-Low Rank Adaptation(Qlora).md",
    "notebooks/.gitkeep",
    "src/finetuning/config/lora_params.yaml",
    "src/finetuning/config/model_loading_params.yaml",
    "src/finetuning/config/trainer_params.yaml",
    "src/finetuning/exception/__init__.py",
    "src/finetuning/logger/__init__.py",
    "src/finetuning/utils/__init__.py",
    "src/finetuning/applying_lora.py",
    "src/finetuning/data_formatting.py",
    "src/finetuning/data_preparation.py",
    "src/finetuning/demo.py",
    "src/finetuning/inference_testing.py",
    "src/finetuning/merge_base_and_finetuned_model.py",
    "src/finetuning/model_and_tokenizer_pusher_to_s3.py",
    "src/finetuning/model_loader.py",
    "src/finetuning/model_trainer.py",
    "src/finetuning/training_config.py",
    ".gitignore",
    "demo.py",
    "LICENSE",
    "README.md",
    "requirements.txt",
    "setup.py",
    "template.py",
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
