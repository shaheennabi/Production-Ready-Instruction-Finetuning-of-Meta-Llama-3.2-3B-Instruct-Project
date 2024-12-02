from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from src.finetuning.model_loader import ModelLoader
from src.finetuning.data_formatting import DataFormatting
from src.finetuning.applying_lora import ApplyLora
from src.finetuning.utils import load_yaml_config

class Trainer:
    def __init__(self):
        try:
            logging.info("Initializing Trainer class.")
            
            self.model_loader = ModelLoader()
            logging.info("ModelLoader initialized.")
            
            self.quantized_model, self.tokenizer = self.model_loader.initiate_model_loader()
            logging.info("Model and tokenizer loaded successfully.")
            
            self.apply_lora = ApplyLora()
            self.lora_layers_and_quantized_model = self.apply_lora.initiate_lora()
            logging.info("LoRA layers applied successfully.")
            
            self.data_formatting = DataFormatting()
            self.formatted_dataset = self.data_formatting.initiate_data_formatting()
            logging.info("Dataset formatting completed.")
            
            # Load parameters from YAML
            config = load_yaml_config("src/finetuning/config/trainer_params.yaml")
            self.params = config.get('args', {})
            logging.info("Configuration loaded from YAML.")
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise CustomException(e)

    def trainer(self):
        try:
            logging.info("Setting up SFTTrainer.")
            
            trainer = SFTTrainer(
                model=self.lora_layers_and_quantized_model,
                tokenizer=self.tokenizer,
                train_dataset=self.formatted_dataset,
                dataset_text_field="text",
                max_seq_length=2048,
                data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
                dataset_num_proc=2,
                packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=self.params['per_device_train_batch_size'],
                    gradient_accumulation_steps=self.params['gradient_accumulation_steps'],
                    warmup_steps=self.params['warmup_steps'],
                    max_steps=self.params['max_steps'],
                    learning_rate=self.params['learning_rate'],
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=self.params['logging_steps'],
                    optim=self.params['optim'],
                    weight_decay=self.params['weight_decay'],
                    lr_scheduler_type=self.params['lr_scheduler_type'],
                    seed=self.params['seed'],
                    output_dir=self.params['output_dir']
                )
            )
            logging.info("SFTTrainer setup successfully.")
            return trainer
        except Exception as e:
            logging.error(f"Error during SFTTrainer setup: {str(e)}")
            raise CustomException(e)

    def initiate_trainer(self):
        try:
            logging.info("Starting Trainer initiation.")
            return self.trainer()
        except Exception as e:
            logging.error(f"Error during Trainer initiation: {str(e)}")
            raise CustomException(e)
