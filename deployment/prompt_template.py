import os
from deployment.logger import logging
from deployment.exception import CustomException
from transformers import AutoTokenizer, AutoModelForCausalLM

class PromptTemplate:
    """
    A class for generating responses using a fine-tuned language model loaded from a local directory.
    """

    def __init__(self, model_dir):
        """
        Initializes the PromptTemplate by loading the model and tokenizer from a local directory.

        Args:
            model_dir (str): Path to the directory where the model and tokenizer are saved.
        """
        try:
            # Ensure the model directory exists
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")

            logging.info(f"Loading model and tokenizer from {model_dir}.")

            # Load tokenizer and model from the directory
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.finetuned_model = AutoModelForCausalLM.from_pretrained(model_dir)

            logging.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logging.error(f"Error during model/tokenizer loading: {str(e)}")
            raise CustomException(f"Failed to load model and tokenizer: {str(e)}")

    def generate(self, user_message):
        """
        Generates a response for the given user message.

        Args:
            user_message (str): The input message from the user.

        Returns:
            str: The cleaned and decoded response from the model.

        Raises:
            CustomException: If an error occurs during response generation.
        """
        try:
            logging.info("Generating response for the given message.")

            # Prepare input message in chat template format
            messages = [{"role": "user", "content": user_message}]

            # Tokenize input message
            inputs = self.tokenizer(messages, return_tensors="pt", padding=True, truncation=True).to("cuda")

            # Generate attention mask
            attention_mask = (inputs.input_ids != self.tokenizer.pad_token_id).long()

            # Generate output using the fine-tuned model
            outputs = self.finetuned_model.generate(
                input_ids=inputs.input_ids,
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
                # Remove unwanted formatting from the output
                if "user\n\n" in response:
                    response = response.split("user\n\n")[1]
                if "assistant\n\n" in response:
                    response = response.split("assistant\n\n")[1]
                clean_output.append(response.strip())

            logging.info("Response generation successful.")
            return clean_output

        except Exception as e:
            logging.error(f"Error during response generation: {str(e)}")
            raise CustomException(f"Response generation failed: {e}")
