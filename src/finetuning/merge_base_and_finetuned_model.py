import os
from peft import PeftModel
from src.finetuning.model_trainer import ModelTrainer
from src.finetuning.model_loader import ModelLoader
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException

class MergeModels:

    def __init__(self):
        # Initialize ModelLoader and ModelTrainer
        self.model_loader = ModelLoader()
        self.model_trainer = ModelTrainer()

        # Load the quantized model and tokenizer from ModelLoader
        self.quantized_model, self.tokenizer = self.model_loader.initiate_model_loader()

        # Load the finetuned model and tokenizer from ModelTrainer
        self.finetuned_model, _ = self.model_trainer.initiate_model_trainer()  # No need to use tokenizer here

        logging.info("Models and Tokenizers Loaded Successfully")

    def final_model(self):
        try:
            # Merging the models using PeftModel
            logging.info("Merging the quantized and finetuned models...")
            final_model = PeftModel.from_pretrained(self.quantized_model, self.finetuned_model)
            final_model = final_model.merge_and_upload()

            # Define save directory
            save_dir = "/content/merged_model"
            os.makedirs(save_dir, exist_ok=True)
            logging.info(f"Save directory created at: {save_dir}")

            # Save the merged model and tokenizer
            final_model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

            logging.info("Merged model and tokenizer saved successfully.")

        except Exception as e:
            logging.error(f"An error occurred while merging models: {str(e)}")
            raise CustomException(f"Error while merging models: {str(e)}")

