from unsloth import FastLanguageModel
import torch
from peft import PeftModel
from src.finetuning.model_loader import ModelLoader
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from src.finetuning.utils import load_yaml_config


class ApplyLora:

    def __init__(self):
        # Initialize the ModelLoader and load quantized model
        self.model_loader = ModelLoader()  # Create an instance of ModelLoader
        self.quantized_model, self.tokenizer = self.model_loader.initiate_model_loader()

        # Load LoRA parameters from the YAML file
        config = load_yaml_config("src/finetuning/config/lora_params.yaml")
        self.params = config.get('lora_params', {})

    def apply_lora_layers(self):
        """
        Apply LoRA layers to the quantized model using the loaded parameters.
        :return: lora_layers_and_quantized_model
        """
        try:
            # Apply LoRA to the quantized model using the provided parameters
            lora_layers_and_quantized_model = FastLanguageModel.get_peft_model(
                self.quantized_model,  # Use the loaded quantized model
                r=self.params['r'],
                target_modules=self.params['target_modules'],
                lora_alpha=self.params['lora_alpha'],
                bias=self.params['bias'],
                use_gradient_checkpointing=self.params['use_gradient_checkpointing'],
                random_state=self.params['random_state'],
                use_rslora=self.params['use_rslora']
            )
            logging.info("LoRA layers applied successfully.")
            return lora_layers_and_quantized_model

        except Exception as e:
            logging.error(f"Error applying LoRA layers: {e}")
            raise CustomException(e)

    def initiate_lora(self):
        """
        Wrapper method to initiate LoRA application process.
        :return: lora_layers_and_quantized_model
        """
        return self.apply_lora_layers()
