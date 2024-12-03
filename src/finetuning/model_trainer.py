import os
from unsloth.chat_templates import train_on_responses_only
from src.finetuning.training_config import Trainer
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from src.finetuning.applying_lora import ApplyLora
from src.finetuning.model_loader import ModelLoader

class ModelTrainer:
    def __init__(self):
        try:
            logging.info("Initializing ModelTrainer.")
            
            # Initialize Trainer and get training configuration
            self.trainer = Trainer()
            self.trainer = self.trainer.initiate_trainer()
            logging.info("Training configuration loaded successfully.")
            
            # Initialize LoRA and model components
            self.apply_lora = ApplyLora()
            self.lora_layers_and_quantized_model = self.apply_lora.initiate_lora()
            logging.info("LoRA layers and quantized model initialized successfully.")
            
            # Initialize ModelLoader to get model and tokenizer
            self.model_loader = ModelLoader()
            self.quantized_model, self.tokenizer = self.model_loader.initiate_model_loader()
            logging.info("Quantized model and tokenizer loaded successfully.")
            
            # Create save directory
            self.save_dir = "/content/model"
            os.makedirs(self.save_dir, exist_ok=True)
            logging.info(f"Save directory created at: {self.save_dir}")
        except Exception as e:
            logging.error(f"Error during initialization of ModelTrainer: {str(e)}")
            raise CustomException(e)

    def start_training(self):
        try:
            logging.info("Starting training process.")
            
            # Train using the provided template
            start_training = train_on_responses_only(
                self.trainer,
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
            )
            
            # Begin training and fetch trainer stats
            trainer_stats = start_training.train()
            logging.info(f"Training completed successfully. Trainer stats: {trainer_stats}")
            
            # Save the fine-tuned model
            finetuned_model = self.lora_layers_and_quantized_model.save_pretrained(self.save_dir)
            logging.info("Fine-tuned model saved successfully.")
            
            # Save the tokenizer
            tokenizer = self.tokenizer.save_pretrained(self.save_dir)
            logging.info("Tokenizer saved successfully.")
            
            return finetuned_model, tokenizer
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise CustomException(e)

    def initiate_model_trainer(self):
        try:
            logging.info("Initiating ModelTrainer.")
            return self.start_training()
        except Exception as e:
            logging.error(f"Error during ModelTrainer initiation: {str(e)}")
            raise CustomException(e)
