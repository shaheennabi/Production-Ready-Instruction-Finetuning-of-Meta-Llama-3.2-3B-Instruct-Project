# 🎋🌿 **Production-Ready Instruction Fine-Tuning of Meta LLaMA 3.2 3B Instruct Project** 🌿🎉  

## **Problem Statement**  
---  
*Note: This project simulates an industry-standard scenario where I am assuming the role of a developer at XYZ Company. The LLaMA 3.2 (3B) model has been successfully deployed in production as part of our product. However, to better serve our large user base of Kannada speakers, fine-tuning the model on a Kannada-specific conversation dataset has become essential.*  




**High-Level Note:** All detailed information about **Instruction Fine-Tuning**, concepts like **reward models**, **training of large language models**, **Reinforcement Learning from Human Feedback (RLHF)**, **Direct Preference Optimization**, **Parameter-Efficient Fine-Tuning**, **Higher Floating-Point Precision Conversion to Lower Precision**, **Quantization**, **LoRA**, and **QLoRA** have already been covered in the `docs/` folder of this repository. Please refer to the documentation there for an in-depth understanding of how these concepts work. In this readme, I will go with project pipeline etc or the main details relevant to the project.


---  


# 🌟 **Fine-Tuning LLaMA 3.2 3B for Kannada Language Adaptation** 🌟  

At **XYZ Company**, we adopted the **LLaMA 3.2 (3B) model** as the core **AI foundation for our product** to provide cutting-edge AI-driven solutions. However, due to our **large Kannada-speaking user base**, the model required fine-tuning to better cater to their needs. After analyzing its performance, our **manager decided** that fine-tuning on a Kannada-specific dataset was essential to enhance the model’s effectiveness.


To achieve this, we are leveraging the **Hugging Face dataset** `charanhu/kannada-instruct-dataset-390k`, containing **390,000 high-quality rows of Kannada instructions**. This dataset serves as the foundation for fine-tuning the model to:  
-  **Better understand Kannada**: Improve comprehension of the language’s syntax, semantics, and nuances.  
-  **Generate accurate responses**: Ensure the model aligns with Kannada-speaking users' expectations and use cases.  
-  **Enhance the overall user experience**: Build a model that feels intuitive and responsive to Kannada-related queries.  




### **My Role as a Developer** 🎋  

As a developer, I am responsible for delivering a Instruction fine-tuned **LLaMA 3.2 3B** model that aligns with the defined **Key Performance Indicator (KPI)** objectives and ensures exceptional performance for Kannada-speaking users.  

- I will **instruct fine-tune** the model using the high-quality **Kannada dataset** from **Hugging Face** (`charanhu/kannada-instruct-dataset-390k`).  

- To address the constraints of **limited GPU resources**, I will implement **QLoRA-based 4-bit precision quantization** using **Unsloth**, which involves:  
  - First **quantizing the model** to 4-bit precision to reduce computational overhead.  
  - Adding **LoRA (Low-Rank Adaptation) layers** to fine-tune the model efficiently within **Google Colab**, ensuring optimal resource utilization without compromising performance.  

- This project is being executed under a **tight deadline**, which requires a strategic focus on both **efficiency and quality**.  

I will collaborate closely with the **AI Systems Team** and **prompt engineers** to ensure the fine-tuning process adheres to business objectives and meets user requirements.  




---
### **Project Goals** 🎯  

1. **Serve Our Large Kannada-Speaking Customer Base**  
   - Adapt the **LLaMA 3.2 3B** model to effectively understand and respond to queries from our extensive Kannada-speaking audience, ensuring an improved and localized user experience.  

2. **Enhance Kannada Language Understanding**  
   - Fine-tune the model using the **Hugging Face Kannada dataset** (`charanhu/kannada-instruct-dataset-390k`), focusing on instruction-specific tasks to ensure accurate, context-aware, and culturally relevant responses in Kannada.  

3. **Optimize Resource Utilization with Google Colab**  
   - Use **Unsloth** to load the model and **QLoRA-based 4-bit precision quantization** for fine-tuning, leveraging **Google Colab** to minimize resource consumption while delivering high-quality results.  




---
## **My Approach** 🚀  

The **instruct-based fine-tuning** process will adhere to industry standards, ensuring the model is perfectly tested after training.  

### **Steps in My Approach**  

1. **Dataset Preparation**  
   - Use the **Hugging Face Kannada dataset** (`charanhu/kannada-instruct-dataset-390k`) for training, focusing on enhancing the model's performance in understanding and responding to Kannada-specific queries.  

2. **Efficient Training with Quantization**  
   - Optimize the training process by implementing **4-bit precision quantization** using **QLoRA** for efficient resource utilization.  
   - Leverage **Google Colab's limited GPU resources** to achieve faster training without compromising the quality of the fine-tuning process.  

3. **Model Deployment and Feedback Loop**  
   - Once the model is fine-tuned, it will be uploaded to an **S3 bucket** with **tokenizer** for easy access for deployment team.  

This approach ensures a resource-efficient, scalable, and production-ready model tailored to meet the needs of Kannada-speaking users.  


**Note: This is high-level view.**


![overview](https://github.com/user-attachments/assets/7b743e25-35b1-4c1e-a48c-20c62fc117ab)




--- 

## **Challenges Encountered** 🎋  

The project faced several challenges, including:  

- **Limited GPU Resources**: Fine-tuning a large model was challenging due to the scarcity of available GPU resources.  
- **Timeline Constraints**: A tight project timeline, driven by the large user base, required rapid action and attention.  



## **How I Fixed Challenges** 🌟  

- To address **GPU limitations**, I utilized **Google Colab** with **4-bit precision quantization** to enable efficient fine-tuning within the available resource constraints.  

- I worked closely with **prompt engineers** to accelerate the fine-tuning process, ensuring we met the project deadline despite the tight timeline.  
  


---

###  Project System Design (or pipeline) 🎋🌿

### **Finetuning Pipeline**  
- The **finetuning process** will be executed only once for this project.  
- **Quantization using `unsloth`:** The model is quantized to 4-bit precision, optimizing it for faster and more efficient finetuning.  
- **Fine-tuning LoRA layers:** These are trained in 16-bit precision for better accuracy. After fine-tuning, the LoRA layers are merged back into the quantized model.  
- Once fine-tuning is complete, the **merged model** along with the tokenizer is uploaded to an **S3 bucket**. This provides a centralized storage location and ensures that the model and tokenizer are ready for deployment or future use.  

  - While this modular structure is prepared for scalability, **for this project**, the fine-tuning is executed in a **Colab-based Jupyter Notebook**. This is because the computational requirements of fine-tuning necessitate the use of external GPU resources available in Colab. From this experimental notebook, the fine-tuned model and tokenizer are pushed directly to S3.  
  - The modular code in `src/finetuning` ensures that if fine-tuning is required again in the future, any developer can easily understand and reuse the logic by running the code independently.  



## Now let's Talk about the Pipeline  🚀

*This is the diagram, of how the pipeline will look:*

![Finetuning Pipeline](https://github.com/user-attachments/assets/bc09764b-b5a1-4614-b872-cc6d9cd88bdc)


*Note: Fine-tuning code will be entirely modular, but I have used **Google Colab** for training, if you have high-end machine make sure you execute **pipeline** in modular fashin*

## Fine-tuning Pipeline 💥
**Note:** The fine-tuning pipeline code is modularized in the `src/finetuning` folder of this repository. If you have access to **high-performance resources** like AWS SageMaker or high-end GPUs, you can execute the modularized files in sequence: start with the **Trainer** to fine-tune the model, then proceed to **Inference** for generating predictions, followed by the **Merge Models** file to combine the fine-tuned model with the base model, and finally, use the **Push to S3** script to upload the final model and tokenizer to your S3 bucket. However, if you lack access to higher-end GPUs or a cloud budget, I recommend using **Google Colab's free tier**. In this case, skip the modularized part and directly execute the provided Jupyter Notebook inside `notebooks/` to fine-tune the model, then upload the `model` and `tokenizer` directly to S3 from the Colab notebook. **Caution:** The modularized pipeline has not been tested thoroughly because I do not have access to high-end compute resources. If you encounter issues while running the pipeline, please raise an issue in the repository, and I will address it immediately.

---
### Installing the required libraries
* Unsloth gives a lot of issues while installing, so execute these code cells one by one in sequence to avoid any problems.

````bash
# Run this first (cell 1)
!python -m pip install --upgrade pip
!pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install xformers[torch2]  # Install xformers built for PyTorch 2.x
!pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
!pip install "git+https://github.com/huggingface/transformers.git"
!pip install trl
!pip install boto3
````

```bash
# Run this cell (cell 2)
!pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Upgrade PyTorch to a compatible version
!pip install xformers  # Install xformers after upgrading PyTorch
```

```bash
# cell 3
!pip uninstall torch torchvision torchaudio -y  # Uninstall existing PyTorch, torchvision, and torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Install PyTorch, torchvision, and torchaudio with CUDA 11.8
```

```bash
# cell 4
!pip uninstall xformers -y
!pip install xformers[torch2]  # Install xformers built for PyTorch 2.x
```

### Importing Necessary Libraries

<img width="656" alt="Importing Necessary Libraries" src="https://github.com/user-attachments/assets/dfb4fdee-0513-4202-b5d1-167e15689354">


- **FastLanguageModel**: Fine-tuned in 4-bit precision for optimized performance and reduced memory usage.
- **SFTTrainer**: Efficiently handles the training process with large models.
- **AutoModelForCausalLM & AutoTokenizer**: Automatically load the pre-trained model and tokenizer for causal language tasks.
- **TrainingArguments**: Configures training settings such as batch size and learning rate.
- **Torch**: Powers the training process using PyTorch.
- **Datasets**: Used for dataset loading and processing.
- **PeftModel** It is used to apply techniques like LoRA to pre-trained models, enabling task-specific adaptations with fewer trainable parameters.



###  Loading the Model

<img width="640" alt="Loading  Model" src="https://github.com/user-attachments/assets/89013450-1bb1-4a29-9ad4-2a620004064e">


- **`max_seq_length`**: Specifies the maximum token length for inputs, set to 2048 tokens in this case.
- **`dtype`**: Auto-detects the optimal data type for model weights, typically `float32` or `float16`.
- **`load_in_4bit`**: Enables 4-bit quantization, reducing memory usage while maintaining model performance.
- **`model_name`**: `unsloth/Llama-3.2-3B-Instruct`, which will be used for fine-tuning and is sourced from Unsloth.

*We obtain the **quantized_model** and **tokenizer** by passing these parameters into **FastLanguageModel.from_pretrained**.*







### Applying  Lora layers

<img width="620" alt="Applying  Lora" src="https://github.com/user-attachments/assets/062a2115-d24d-4ede-9c83-2fc9665cdaa1">

- **`r`**: LoRA rank, set to `16`, determines the size of the low-rank adaptation.
- **`target_modules`**: Specifies the layers in the model to which LoRA should be applied, including `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- **`lora_alpha`**: Scaling factor for LoRA layers, set to `16` for efficient weight updates.
- **`lora_dropout`**: Dropout rate for LoRA layers, set to `0` here for optimization.
- **`bias`**: Indicates whether additional bias terms should be added; set to `"none"` for simplicity.
- **`use_gradient_checkpointing`**: Uses Unsloth’s gradient checkpointing method to reduce memory usage during training.
- **`random_state`**: Sets the random seed for reproducibility, here set to `3407`.
- **`use_rslora`**: Rank stabilized LoRA, disabled here (`False`) but can be enabled for increased stability.

*The **lora_layers_and_quantized_model** are obtained by applying these parameters through the **FastLanguageModel.get_peft_model** function.*








### Data Preparation

<img width="920" alt="Dataset Preparation" src="https://github.com/user-attachments/assets/869f6569-df05-455f-bd7e-ba71dc036593">

- **Loading Dataset**:  
  - The Kannada Instruct Dataset is loaded using **`load_dataset`** from the `datasets` library.  
  - **Dataset Source**: `"charanhu/kannada-instruct-dataset-390-k"`.  
  - **Split**: The training split is used (`split="train"`).

- **Standardizing Dataset**:  
  - The `standardize_sharegpt` function from **`unsloth.chat_templates`** is applied to prepare the dataset for training.  
  - **Purpose**: Ensures the dataset aligns with ShareGPT-like formatting standards, making it compatible for conversational AI fine-tuning.  
  - **Key Benefits**:  
    - Cleans and structures the dataset for consistency.  
    - Maps raw inputs and outputs into an organized format (e.g., system messages, user queries, and assistant responses).  
    - Reduces preprocessing overhead during model fine-tuning.

- **Inspecting Data**:  
  - A loop is used to print the first item of the dataset to understand its structure and verify standardization.  









### Data Formatting(what model expects for instruction tuning)

<img width="920" alt="Prompt Formatting" src="https://github.com/user-attachments/assets/58f7c5cf-945a-43d7-a9cf-670eee3261e6">

- **Formatting Data Using Tokenizer**:  
  - A function **`formatting_prompts_func`** is defined to format the dataset's conversation data.
  - **Key Details**:  
    - Uses the tokenizer **indirectly** to format conversations but does not perform tokenization.  
    - The function applies **`tokenizer.apply_chat_template`** to each conversation, ensuring proper formatting for fine-tuning.  
    - **Parameters in `apply_chat_template`**:  
      - `tokenize=False`: Ensures the data is not tokenized but only formatted.  
      - `add_generation_prompt=False`: Disables automatic generation prompts for clean data formatting.  

- **Mapping Formatting Function to Dataset**:  
  - **`dataset.map`** is used to apply the formatting function (`formatting_prompts_func`) to the entire dataset in batches.  
  - **Output**: A new key **`text`** in the dataset containing the formatted conversation strings.

- **Inspecting Formatted Data**:  
  - A loop is used to print the first formatted item of the dataset to verify the results.  







###  Training Configurations

<img width="614" alt="Training Configuration" src="https://github.com/user-attachments/assets/956acc04-ac6f-497b-9c12-9cc33b70301b">

- **Initializing Fine-Tuning with SFTTrainer**:  
  - **Purpose**: Fine-tuning the model by training LoRA layers while keeping the quantized base model frozen.  

- **Key Components**:  
  - **`model`**:  
    - Contains the quantized base model with LoRA layers for efficient parameter updates.  
    - Only LoRA layers are trainable; the base model remains static.  
  - **`tokenizer`**: Used to preprocess input data into a format compatible with the model.  
  - **`train_dataset`**: The dataset to fine-tune the model, here set to the formatted **`dataset`**.  
  - **`dataset_text_field`**: Specifies the field in the dataset containing formatted text data (key: **`text`**).  
  - **`max_seq_length`**: Limits tokenized input sequences to 2048 tokens.  
  - **`data_collator`**:  
    - Prepares batches for training using **`DataCollatorForSeq2Seq`**, ensuring compatibility with sequence-to-sequence tasks.  
  - **`dataset_num_proc`**: Sets parallel processing to **2 threads** for efficiency during data preparation.  
  - **`packing`**: Disables input packing to keep data unaltered.  

### **`TrainingArguments`** Parameters:

- **`per_device_train_batch_size`**:  
  - Defines the number of training samples processed simultaneously on each GPU or CPU.  
  - In this case, **4 samples per device**, which means if multiple GPUs are used, the total effective batch size will be multiplied by the number of GPUs.  

- **`gradient_accumulation_steps`**:  
  - Accumulates gradients over **4 mini-batches** before performing a single optimizer step.  
  - This allows for the simulation of a larger batch size while using less memory, effectively making the batch size = `per_device_train_batch_size × gradient_accumulation_steps`.  
  - Example: Here, the effective batch size becomes **4 × 4 = 16**.  

- **`warmup_steps`**:  
  - Gradually increases the learning rate over **20 steps** at the beginning of training.  
  - Prevents sudden large updates to weights, stabilizing training and reducing the risk of exploding gradients.  

- **`max_steps`**:  
  - Specifies the **maximum number of training steps**.  
  - Training will terminate after completing **300 steps**, regardless of the number of epochs completed.  

- **`learning_rate`**:  
  - Controls the rate at which model weights are updated.  
  - A smaller value like **1.5e-4** ensures slow and stable convergence, especially critical for fine-tuning large models.  

- **`fp16` and `bf16`**:  
  - **`fp16`**: Mixed-precision training using 16-bit floating-point numbers, which speeds up training and reduces memory usage.  
  - **`bf16`**: Alternative to `fp16`, supported on newer hardware like **NVIDIA A100 GPUs**, with better numerical stability.  
  - **Logic**: If the system supports **`bfloat16`**, it will use it; otherwise, it defaults to **`fp16`**.  

- **`logging_steps`**:  
  - Logs metrics (e.g., loss, learning rate) every **10 steps**, helping monitor training progress.  

- **`optim`**:  
  - Specifies the optimizer used for weight updates, here **`adamw_8bit`**, which is a memory-efficient version of the Adam optimizer.  
  - Suitable for training large models with reduced memory usage while maintaining performance.  

- **`weight_decay`**:  
  - Applies a regularization penalty of **0.02** to model weights, helping prevent overfitting.  

- **`lr_scheduler_type`**:  
  - Adjusts the learning rate dynamically during training.  
  - **`linear` scheduler**: Decreases the learning rate linearly from its initial value to zero as training progresses.  

- **`seed`**:  
  - Sets the random seed to **3407** for ensuring reproducibility.  
  - Fixes randomness in data shuffling, weight initialization, and other stochastic processes.  

- **`output_dir`**:  
  - Specifies the directory where training outputs (e.g., model checkpoints, logs) are saved.  
  - Example: All artifacts will be stored in the folder **`outputs`**.  

- **Final Output**:  
  - The **`trainer`** object manages the training loop, including data preprocessing, forward/backward passes, and logging.  
  - Fine-tunes LoRA layers to enhance the model's performance on the provided dataset.  











### Model Training

<img width="856" alt="Model  Training" src="https://github.com/user-attachments/assets/075ee343-8412-4ad4-bb4b-dd569663c4fd">

- **`train_on_responses_only`**: This function from `unsloth.chat_templates` modifies the training loop to specifically focus on the model's responses, excluding the instructions. This technique allows the model to better specialize in generating responses rather than understanding instructions, which can be useful in fine-tuning models for tasks like dialogue generation or question answering.

- **`instruction_part`**: Specifies the tokenized start and end markers for the user instruction. This helps to differentiate the instruction from the response, so the model learns to ignore the instruction when fine-tuning and focus only on generating the appropriate response.

- **`response_part`**: Specifies the tokenized start and end markers for the model's response. By isolating the response, the model is encouraged to generate responses that align with the given instruction but is not directly trained on the instruction itself during the process.

- **`trainer.train()`**: Initiates the training process on the dataset, where the model is specifically trained on generating accurate responses while the instructions are handled separately. This helps improve the quality of responses in tasks where the model needs to generate coherent replies based on the input conversation context.

Here in this approach, I performed instruction fine-tuning, but with a primary emphasis on **response generation** rather than directly interpreting the instruction itself. While the model is still trained to follow instructions, the core training goal is to enhance its ability to generate **contextually relevant, coherent, and accurate responses** based on the given instructions.

The reason for focusing more on responses is to make the model **more dynamic and conversational**, ensuring that it generates high-quality outputs even when the instructions are varied or complex. Instead of explicitly focusing on how well the model understands the instructions, the priority is on improving its response generation, making it better at delivering useful, human-like answers.

In this methodology, although the model is still guided by instructions (e.g., "Summarize the paragraph"), the primary focus is placed on **optimizing response generation**. The model is fine-tuned to produce **fluent, accurate, and contextually relevant responses**, ensuring it generates outputs that align with human expectations and preferences, whether the response is concise, natural, or creative.







### Inference

<img width="713" alt="Inference  1" src="https://github.com/user-attachments/assets/189c2d17-9026-4cb3-bdfb-95435b075fae">

<img width="901" alt="Inference 2" src="https://github.com/user-attachments/assets/ea31462b-9e1c-4575-9120-5390cfbc23e2">


#### Steps for Inference

 **Prepare the Model for Inference**:  
   The fine-tuned model is loaded using `FastLanguageModel.for_inference`, ensuring compatibility with Unsloth's inference pipeline.

 **Define User Inputs**:  
   Input messages are defined explicitly to avoid unnecessary system messages.  
   **Example Input**:  
   `"ಪರಿಸರದ ಬಗ್ಗೆ ಬರೆಯಿರಿ ಮತ್ತು ಪ್ರಬಂಧವನ್ನು ಬರೆಯಿರಿ."` (Write an essay about the environment.)

 **Tokenization and Formatting**:  
   The input is tokenized using the `tokenizer` with the following options:
   - **`tokenize=True`**: To convert text into tokens.
   - **`add_generation_prompt=True`**: Ensures generation starts from the assistant's perspective.
   - **`return_tensors="pt"`**: Outputs PyTorch tensors for model compatibility.

 **Generating Responses**:  
   The fine-tuned model generates a response with:
   - **`max_new_tokens=1024`**: Defines the maximum number of tokens in the output.
   - **`temperature=1.5`**: Adds randomness to the output for creative generation.
   - **`min_p=0.1`**: Filters out less probable tokens to improve relevance.

 **Decoding and Post-Processing**:  
   Outputs are decoded and cleaned by removing unwanted metadata or system messages. This ensures the response is concise and focused.

 **Output Example**:  
   The generated response aligns with the instruction, such as providing a detailed essay on the environment in the Kannada language.













### Saving the Model & Tokenizer

<img width="453" alt="Saving the model and tokenizer" src="https://github.com/user-attachments/assets/f6eb0858-f51e-452d-a65b-83945537e487">

- **Save Directory**: Defines a directory to store the model and tokenizer. Creates it if it doesn’t exist.  
- **Model Saving**: Saves the fine-tuned LoRA layers and quantized model using `save_pretrained`.  
- **Tokenizer Saving**: Saves the tokenizer to ensure compatibility during inference.  
- **Output Confirmation**: Prints the save path to verify successful storage.
- 


### Merging base model & finetuned lora layers

<img width="557" alt="Merge base model and finetuned layers" src="https://github.com/user-attachments/assets/15d66a2b-dfb9-471c-8fe0-9b13640d45e4">

- **Base Model Setup**:  
  - Loads a base model (`unsloth/Llama-3.2-3B-Instruct`) with **4-bit quantization** to reduce memory usage.  
  - Sets the **maximum sequence length** to 2048 for handling long inputs.  

- **Fine-Tuned Weights Integration**:  
  - Loads fine-tuned LoRA weights from the specified path.  
  - Merges LoRA weights into the base model using `merge_and_unload`, ensuring a fully integrated model with no residual adapter layers.  

- **Saving the Final Model and Tokenizer**:  
  - Saves the merged model and tokenizer to a specified directory.  

**Save Location**: `/content/merged_model`












### Pushing Model & Tokenizer to S3 Bucket


<img width="923" alt="s3 1" src="https://github.com/user-attachments/assets/76f0d76f-b270-46c1-a2c9-b1d7232b0b72">

<img width="911" alt="s3 2" src="https://github.com/user-attachments/assets/a274524a-b9cc-4f07-addc-c7c434f4f0b9">

<img width="909" alt="s3 3" src="https://github.com/user-attachments/assets/b09ff2f4-da2e-492f-8586-269ce2a29ea8">




- **AWS Credentials Setup**:  
  - Environment variables are configured for **AWS Access Key**, **Secret Key**, and **Region** to enable secure access to AWS services.  

- **S3 Client Initialization**:  
  - Configures `boto3` to interact with S3 using the specified credentials.  

- **Specify Local and S3 Paths**:  
  - **Local Path**: `/content/merged_model` (contains fine-tuned model and tokenizer files).  
  - **S3 Bucket**: `instruct` with a folder prefix `files/` to organize uploads.  

- **Selective File Upload**:  
  - Only uploads `model.safetensors` and `tokenizer.json` files, ensuring other files are skipped.  



---
## Guide for Developers 🌿🎇✨💚🎆🌱🎇✨💚🎆 


### Fine-tuning 🌿

Dear **developers** if you are looking to build a similar project, I recommend using Google Colab as your primary environment for training and fine-tuning. Colab provides free access to GPUs (like T4 or P100) which can help speed up the process. For efficient fine-tuning, consider using PEFT (Parameter Efficient Fine-Tuning) techniques like LoRA, which only updates a subset of the model's parameters, reducing memory usage and computational cost. You can load pre-trained models and fine-tune them in 4-bit precision, which makes training more resource-efficient. Be sure to format your dataset according to the Kannada Instruct dataset format for instruction-based tasks.


### Tree Structure 🌱

```bash
├── PRODUCTION-READY-INSTRUCTION-FINETUNING-OF-META-Llama-3.2-3B Instruct
├── .github/
│   └── FUNDING.yml
├── docs/
│   ├── 1. Understanding Instruction Finetuning.md
│   ├── 2. reward_model.md
│   ├── 3. RLHF with PPO.md
│   ├── 4. Direct Preference Optimization.md
│   ├── 5. Understanding ULMA.md
│   ├── 6. Parameter Efficient Finetuning.md
│   ├── 7. Low Rank Adaptation(LORA).md
│   └── 8. Quantized-Low Rank Adaptation(Qlora).md
├── flowcharts/
│   ├── Finetuning Pipeline.jpg
│   └── overview.jpg
├── log/
│   └── timestamp(log)
├── notebooks/
│   └── Instruct_Tuning_Llama3.2-3B_instruct.ipynb
├── src/finetuning/
│   ├── config/
│   │   ├── lora_params.yaml
│   │   ├── model_loading_params.yaml
│   │   └── trainer_params.yaml
│   ├── exception/
│   │   └── __init__.py
│   ├── logger/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   ├── applying_lora.py
│   ├── data_formatting.py
│   ├── data_preparation.py
│   ├── demo.py
│   ├── inference_testing.py
│   ├── merge_base_and_finetuned_model.py
│   ├── model_and_tokenizer_pusher_to_s3.py
│   ├── model_loader.py
│   ├── model_trainer.py
│   └── training_config.py
├── .gitignore
├── demo.py
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── template.py


```


Happy coding and finetuning! 🎉💚

🎆💚✨🎉🎇💚🎆✨

---

## **License 📜✨**  
This project is licensed under the MIT License.  
You are free to use, modify, and share this project, as long as proper credit is given to the original contributors.  
For more details, check the LICENSE file. 🏛️
