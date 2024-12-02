from unsloth import FastLanguageModel
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from src.finetuning.utils import load_yaml_config


class ModelLoader:
    def __init__(self):
        """
        Initialize the ModelLoader with parameters loaded from the YAML file.
        The parameters are fetched directly from the config file.
        """
        # Load the model_loading_params from the YAML file when the class is instantiated
        config = load_yaml_config("src/finetuning/config/model_loading_params.yaml")
        self.params = config.get('model_loading_params', {})

        # Log initialization
        logging.info("ModelLoader initialized with the following parameters:")
        for key, value in self.params.items():
            logging.info(f"{key}: {value}")

    def load_model(self):
        """
        Load the quantized model and tokenizer using the parameters loaded from the YAML file.
        :return: quantized_model, tokenizer
        """
        try:
            # Ensure that required params are present
            if not all(key in self.params for key in ['max_seq_length', 'dtype', 'load_in_4bit']):
                raise ValueError("Missing one or more required parameters in model_loading_params.")
            
            quantized_model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Llama-3.2-3B-Instruct",
                max_seq_length=self.params['max_seq_length'],
                dtype=self.params['dtype'],
                load_in_4bit=self.params['load_in_4bit']
            )
            logging.info("Model and tokenizer loaded successfully.")
            return quantized_model, tokenizer
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise CustomException(e)

    def initiate_model_loader(self):
        """
        A wrapper method to initiate the model loader process.
        :return: quantized_model, tokenizer
        """
        return self.load_model()
