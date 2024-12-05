from flask import Flask, request, render_template
from unsloth import FastLanguageModel
import torch
import os

app = Flask(__name__)

# Define paths to the local model and tokenizer
model_path = "/home/ubuntu/models/model.safetensors"
tokenizer_path = "/home/ubuntu/models/tokenizer.json"

# Ensure that model and tokenizer files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

# Load the model and tokenizer
finetuned_model = FastLanguageModel.for_inference(model_path)
tokenizer = torch.load(tokenizer_path)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        prompt = request.form["prompt"]

        # Prepare the input messages
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        input_ids = inputs
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # Generate output using the model
        outputs = finetuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            use_cache=True,
            temperature=1.5,
            min_p=0.1
        )

        # Decode and clean the output
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        clean_output = []
        for response in decoded_output:
            response = response.split("user\n\n")[1] if "user\n\n" in response else response
            response = response.split("assistant\n\n")[1] if "assistant\n\n" in response else response
            clean_output.append(response)

        result = clean_output[0]  # Display the first result

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
