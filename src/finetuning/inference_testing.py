import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from src.finetuning.model_trainer import ModelTrainer
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException

class Inference:
    def __init__(self):
        try:
            logging.info("Initializing Inference class.")
            self.model_trainer = ModelTrainer()
            self.finetuned_model, self.tokenizer = self.model_trainer.initiate_model_trainer()
            logging.info("Model and tokenizer successfully loaded for inference.")
        except Exception as e:
            logging.error(f"Error during initialization of Inference: {str(e)}")
            raise CustomException(e)

    def generate(self, user_message):
        try:
            logging.info("Generating response for the given message.")
            
            # Prepare input message
            messages = [{"role": "user", "content": user_message}]
            
            # Tokenize input message
            inputs = self.tokenizer.apply_chat_template(
                messages,  
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
            
            # Generate attention mask
            attention_mask = (inputs != self.tokenizer.pad_token_id).long()

            # Generate output from the fine-tuned model
            outputs = self.finetuned_model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                use_cache=True,
                temperature=1.5,
                top_p=0.1,  
            )

            # Decode and clean up the generated output
            decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            clean_output = []

            for response in decoded_output:
                response = response.split("user\n\n")[1] if "user\n\n" in response else response
                response = response.split("assistant\n\n")[1] if "assistant\n\n" in response else response
                clean_output.append(response)

            # Print the cleaned output
            logging.info("Response generation successful.")
            print(clean_output)
        except Exception as e:
            logging.error(f"Error during response generation: {str(e)}")
            raise CustomException(e)
