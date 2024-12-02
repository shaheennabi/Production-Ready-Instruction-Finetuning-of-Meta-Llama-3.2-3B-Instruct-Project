from unsloth.chat_templates import get_chat_template
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from src.finetuning.data_preparation import DataPreparation
from src.finetuning.model_loader import ModelLoader


class DataFormatting:
    def __init__(self):
        """
        Initializes the DataFormatting class by loading the model, tokenizer, 
        and standardized dataset.
        """
        try:
            # Load the model and tokenizer
            self.model_loader = ModelLoader()
            self.quantized_model, self.tokenizer = self.model_loader.initiate_model_loader()

            # Prepare the standardized dataset
            self.data_prep = DataPreparation()
            self.standardized_dataset = self.data_prep.initiate_datapreparation()

            logging.info("DataFormatting initialization successful.")
        except Exception as e:
            logging.error(f"Error during DataFormatting initialization: {e}")
            raise CustomException(f"Error during DataFormatting initialization: {e}")

    def formatting_prompts_func(self, examples):
        """
        Formats prompts in the dataset by applying a chat template.
        :param examples: A batch of examples from the dataset.
        :return: A dictionary containing formatted text.
        """
        try:
            convos = examples["conversations"]
            texts = [self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                for convo in convos
            ]
            return {"text": texts}
        except Exception as e:
            logging.error(f"Error in formatting prompts: {e}")
            raise CustomException(f"Error in formatting prompts: {e}")

    def apply_formatting(self):
        """
        Applies the formatting function to the standardized dataset.
        :return: The dataset with formatted prompts.
        """
        try:
            formatted_dataset = self.standardized_dataset.map(
                self.formatting_prompts_func, batched=True
            )
            logging.info("Prompt formatting applied successfully.")
            return formatted_dataset
        except Exception as e:
            logging.error(f"Error during dataset formatting: {e}")
            raise CustomException(f"Error during dataset formatting: {e}")

    def initiate_data_formatting(self):
        """
        Initiates the data formatting process.
        :return: The formatted dataset.
        """
        try:
            return self.apply_formatting()
        except Exception as e:
            logging.error(f"Error in data formatting initiation: {e}")
            raise CustomException(f"Error in data formatting initiation: {e}")
