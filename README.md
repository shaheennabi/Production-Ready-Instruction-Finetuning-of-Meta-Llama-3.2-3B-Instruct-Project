# 🎋🌿 **Production-Ready ULMA Instruction Fine-Tuning of Meta LLaMA 3.1 8B Project** 🌿🎉  
*Note: If you are a company training your own language models or a developer practicing fine-tuning, you can adopt this approach to align your model with human preferences effectively. This project incorporates a robust pipeline designed to prevent your model from generating responses that are racist, hateful, or socially harmful.* 
            🚀 Clone this Repository and boom!

## **Problem Statement**  
---  
*Note: This project simulates an industry-standard scenario where I am pretending to work at XYZ Company. The LLaMA 3.1 8B model was deployed in production on our product but was generating harmful outputs such as hate speech, sexually explicit, and racially biased content. This created an immediate need for instruct-based ULMA fine-tuning. As a developer, I approached this problem using industry-standard practices to ensure the model aligns with societal norms and ethical standards while meeting business requirements.*

**High-Level Note:** All detailed information about **Instruction Fine-Tuning**, **ULMA**, concepts like **reward models**, **training of large language models**, **Reinforcement Learning from Human Feedback (RLHF)**, **Direct Preference Optimization**, **Parameter-Efficient Fine-Tuning**, **Higher Floating-Point Precision Conversion to Lower Precision**, **Quantization**, **LoRA**, and **QLoRA** have already been covered in the `docs/` folder of this repository. Please refer to the documentation there for an in-depth understanding of how these concepts work. In this readme, I will go with project pipeline etc or the main details relevant to the project.


---  

At **XYZ Company**, we adopted the **LLaMA 3.1 8B model** as the core **AI foundation for our product**. However, recent **performance evaluations in production environments** exposed critical limitations. These issues led to **customer dissatisfaction** and highlighted the following challenges:  

### **Identified Issues**  
- The model generated **racist text** and **sexual content** in certain scenarios.  
- It produced **hate speech** targeting specific entities, causing potential reputational damage.  
- A significant portion of the generated content was **hateful**, displayed **social norm biases**, and posed **safety risks** to customers.  

### **Need for Immediate Action**  
These challenges raised significant ethical and safety concerns due to the product's large user base. The **AI Systems Team**, under the guidance of management, hired **prompt engineers** to extensively test the model's responses. Despite their efforts, the results remained harmful to customers and society, necessitating a more robust solution.  

### **Proposed Solution: ULMA Instruction Fine-Tuning**  
To address these issues and ensure alignment with societal norms, ethical standards, and human preferences, the team decided to proceed with **instruction-based ULMA fine-tuning**. This solution aims to:  
- Prevent hate speech and harmful content.  
- Reduce biases in model outputs.  
- Ensure safety and ethical compliance in real-world applications.  

Fine-tuning is considered the **only viable method** to deeply align the model with societal and ethical expectations, ensuring it performs reliably and responsibly in production environments.  

### **My Role as a Developer**  🎋
As a developer, I am responsible for delivering a fine-tuned **LLaMA 3.1 8B** model that meets the defined Key Performance Indicator (KPI) objectives while aligning with societal and ethical standards.  

- I will **instruct fine-tune** the model using high-quality **ULMA datasets** from **Anthropic's Hugging Face Dataset** (**Unified-Language-Model-Alignment/Anthropic_HH_Golden**). This process will improve the model’s responses, ensuring it does not generate hateful, sexual, or racist text and aligns with societal norms.  

- Given the constraints of limited GPU resources, I will employ **QLoRA-based 4-bit precision quantization** using **BitsAndBytes**. This involves:  
  - First **quantizing** the model to 4-bit precision.  
  - Adapting **LoRA layers** to fine-tune the model within **Google Colab**, optimizing resource usage without sacrificing performance.  

- Advanced prompting techniques and supervised instruction fine-tuning will be incorporated to maintain robust model performance while balancing potential accuracy trade-offs.  


- This project operates under a **tight deadline**, necessitating a strong focus on efficiency and quality.  

Close collaboration with the **AI Systems Team** and **prompt engineers** will ensure the fine-tuning process meets business objectives and customer requirements, delivering a model that is reliable, ethical, and effective for real-world applications.  



---
## Goals 🎉  

- **Mitigate harmful outputs** such as hate speech, sexually explicit content, and racial bias by instruct-based **ULMA fine-tuning**
- Deliver a **production-ready solution** that aligns the model with societal norms, ethical standards, and customer expectations.  
- **Impress customers** and maintain trust among our **large user base** by ensuring the model generates safe, inclusive, and appropriate content.  
- Prioritize **GPU resource optimization** by leveraging **QLoRA** techniques, to efficiently fine-tune the model.


---
## **My Approach** 🚀

The **Instruct-based ULMA fine-tuning** process will be done as per industry standards, thorough testing will be done before using model for inference.

I will begin by training the model using **Anthropic's HH Golden dataset** to align it with societal norms, human preferences, and to mitigate harmful content such as hate speech, racism, and hallucinations, while improving response accuracy.

To optimize training efficiency, I will implement **quantization** and **4-bit precision** for faster training within **Google Colab**, utilizing limited GPU resources. 

Additionally, I will apply **advanced prompt engineering techniques** to further guide the model’s responses, ensuring it aligns with human preferences and ethical standards, while addressing the key issues identified (e.g., hate speech, racism, and harmful content generation).

Once, model is finetuned It will be pushed to S3 Bucket, and used for inference to test and gather feedback.

**Note: This is high-level view.**



![CI_CD Diagram (2)](https://github.com/user-attachments/assets/ea73009a-26ba-477c-97b9-88672935eb57)



--- 
## Challenges Encountered 🎋

The project encountered several challenges, including:

- **Limited GPU Resources**: Fine-tuning a large model was difficult due to the scarcity of available GPU resources.
- **Human Preferences and Safe Responses**: Ensuring the model generated **accurate responses** without harmful or biased content was a key concern, requiring proper mitigation strategies.
- **Timeline Constraints**: The project timeline posed significant challenges, due to the large user base of the model, requiring quick action and immediate attention.
- **Model Inference on AWS**: Running inference on AWS was costly. This raised concerns regarding both **storage** and **compute costs**.


## How I Fixed Challenges 🌟

- To overcome **GPU limitations**, I used **Google Colab ** with **4-bit precision** for efficient fine-tuning.

- To ensure **human preferences and safe responses**, I used a curated **Anthropic dataset** and applied advanced **prompting techniques** for refinement.

- I collaborated with **prompt engineers** to expedite the fine-tuning process and meet the project deadline.

- For inference, I optimized the model deployment with a **multi-stage Docker setup** (using **Docker Compose**) to reduce image size and improve efficiency.

---
*Note:* *Supervised Instruction* *Unified-Language-Model-Alignment (ULMA)* *has been performed on the* *Llama 3.1 8B parameter model.* *While further improvements could have been achieved even after finetuning using* *Reward Models* *and* *Reinforcement Learning from Human Feedback (RLHF),* *I believe that ULMA instruction fine-tuning is sufficient for this project.* *The* *Llama model* *itself is robust, and my primary focus was on addressing biased or harmful responses by aligning it with ethical and accurate outputs through instruction tuning.* *As I told in the beginning that* **I am pretending that I am working at XYX company** *for going further with* *Reward Model* *and* *RLHF,* *it would require developing a reward model, which involves human experts—a resource often available only to larger companies with dedicated research teams.* *For this problem, instruction tuning effectively meets the requirements.*

---

## Tools and Libraries (Used in This Project) 🎊

### - **accelerate**  
  Used to efficiently distribute and run the training process across hardware (CPU/GPU/TPU), optimizing performance and memory usage.

### - **torch**  
  Core deep learning library utilized for building, training, and evaluating models.

### - **peft**  
  Enables Parameter-Efficient Fine-Tuning (LoRA layers), making fine-tuning large models feasible by training only small additional layers.

### - **bitsandbytes**  
  Facilitates 4-bit quantization of models, reducing memory requirements and enabling efficient handling of large models.

### - **transformers**  
  Provides pre-trained models and tokenizers for easy integration of state-of-the-art NLP models.

### - **trl**  
  Used for reinforcement learning with language models, aiding in fine-tuning for alignment with specific tasks or human preferences.

### - **datasets**  
  A library to seamlessly load and preprocess datasets, including the HuggingFace dataset repository.

### - **google-colab**  
  Platform used for running and experimenting with the fine-tuning pipeline in a cloud-based environment with GPU support.

### - **flask**  
  Lightweight framework for serving the fine-tuned model as an API for inference.

### - **aws**  
  Services like S3 and ECR are used to store and deploy the fine-tuned model and tokenizer, facilitating scalable deployment.


---

###  Project System Design (or pipeline) 🎋🌿
Remember: For this project **Pipeline** is going to be seprated in two different parts

### **1. Finetuning Pipeline**  
- The **finetuning process** will be executed only once for this project.  
- **Quantization using `bitsandbytes`:** The model is quantized to 4-bit precision, optimizing it for faster and more efficient finetuning.  
- **Fine-tuning LoRA layers:** These are trained in 32-bit precision for better accuracy. After fine-tuning, the LoRA layers are merged back into the quantized model.  
- Once fine-tuning is complete, the **merged model** along with the tokenizer is uploaded to an **S3 bucket**. This provides a centralized storage location and ensures that the model and tokenizer are ready for deployment or future use.  
- **Modular Code Structure:**  
  - The **fine-tuning code** is organized under the `src/finetuning` directory.  
  - The directory contains separate files for:  
    - LoRA parameters configuration.  
    - PEFT (Parameter-Efficient Fine-Tuning) setup.  
    - Model loading and initialization logic.  
    - Data ingestion and preprocessing logic.  
  - While this modular structure is prepared for scalability, **for this project**, the fine-tuning is executed in a **Colab-based Jupyter Notebook**. This is because the computational requirements of fine-tuning necessitate the use of external GPU resources available in Colab. From this experimental notebook, the fine-tuned model and tokenizer are pushed directly to S3.  
  - The modular code in `src/finetuning` ensures that if fine-tuning is required again in the future, any developer can easily understand and reuse the logic by running the code independently.  

### **2. Deployment/Inference Pipeline**  
- This pipeline focuses on serving the fine-tuned model for inference and includes:  
  - **Containerization:** The deployment logic, including the Flask API (`app.py`), utility scripts (`inference.py`, `s3_utils.py`), and configuration files (e.g., `requirements.txt`, `.env`), is containerized using Docker.  
  - **Deployment Pipeline:** The Docker image is pushed to **AWS ECR** for deployment. Updates to the deployment logic are handled via **GitHub Actions**, ensuring continuous integration and delivery.  
  - **Model and Tokenizer Retrieval:** During inference, the application fetches the fine-tuned model and tokenizer directly from S3. This ensures modularity and decouples the deployment process from the fine-tuning pipeline.  

### **Why This Modular Approach?**  
1. **Decoupling Finetuning and Deployment:**  
   - The fine-tuning process is resource-intensive and performed only once. By separating it from the deployment pipeline, we avoid unnecessary dependencies.  
2. **Future Scalability:**  
   - The modular structure in `src/finetuning` ensures that developers can independently run and update the fine-tuning logic if needed. For example, if a company or developer with access to high-end hardware wants to fine-tune the model on new data, they can directly use this modular codebase. Finetuning is a one time task so modularization of finetuning is not important, but we can modularize **inference or deployment** part.
3. **Deployment Flexibility:**  
   - The deployment pipeline is designed for continuous updates, allowing enhancements to the inference API, new features, or configuration changes without impacting the fine-tuning code.  

## Now let's Talk about the Fine-tuning Pipeline  🚀

*This is the diagram, of how the pipeline will look:*

![Finetuning Pipeline](https://github.com/user-attachments/assets/86052bbf-a95c-4a67-9242-f6002494d246)


## Fine-tuning Pipeline 💥
---

### 1. **Data Preparation**
We will begin by **ingesting** the data from **HuggingFace**, specifically the dataset **Unified-Language-Model-Alignment/Anthropic_HH_Golden**.  
After ingesting, we will **load** the dataset, and we will transform dataset into **llama instruction** format that llama accepts for finetuning.

![Data Preparation Code](path/to/screenshot1.png)



### 2. **Tokenization**
Using the **LLaMA model's tokenizer**, we will **tokenize** the dataset, ensuring compatibility with the pre-trained model.

![Tokenization Code](path/to/screenshot2.png)



### 3. **Data Splitting**
The data will be split into **training** and **validation** sets.  
The **test set**, already included in the dataset, will be reserved for evaluation after successful fine-tuning of the model.

![Data Splitting Code](path/to/screenshot3.png)



### 4. **Pre-trained Model Loading**
We will **load the pre-trained model** from **HuggingFace** for fine-tuning.

![Model Loading Code](path/to/screenshot4.png)



### 5. **Quantization**
Using **bitsandbytes**, we will convert the model's precision from **32-bit** to **4-bit** to reduce memory requirements and improve efficiency.

![Quantization Code](path/to/screenshot5.png)



### 6. **Save Quantized Model**
The **quantized model** will be saved for comparison with the fine-tuned model later.

![Save Quantized Model Code](path/to/screenshot6.png)



### 7. **PEFT Application**
We will apply **PEFT (LoRA layers)** to the **quantized model**, adding trainable parameters to enable efficient fine-tuning.

![PEFT Code](path/to/screenshot7.png)



### 8. **Fine-tuning**
The model will be fine-tuned on the **training data**, with **validation** and **early stopping** mechanisms to prevent overfitting.

![Fine-tuning Code](path/to/screenshot8.png)


### 9. **Model Merging**
We will merge the **quantized base model** and the **fine-tuned LoRA layers**, combining **4-bit** and **32-bit precision** components.

![Model Merging Code](path/to/screenshot10.png)



### 10. **Evaluation**
The merged model will be evaluated on the **test set** using the **perplexity metric** to measure its performance.

![Evaluation Code](path/to/screenshot11.png)



### 11. **Testing with Prompts**
The model's output will be tested using carefully designed **prompts** to verify alignment with desired behaviors.

![Prompt Testing Code](path/to/screenshot12.png)



### 12. **Model Comparison**
We will compare the **quantized model** and the **fine-tuned model** using the same **prompts** to analyze improvements.

![Model Comparison Code](path/to/screenshot13.png)



### 13. **Advanced Prompting**
Advanced **prompting techniques** will be applied to further guide the model's responses and evaluate its alignment with human preferences.

![Advanced Prompting Code](path/to/screenshot14.png)



### 14. **Artifact Upload**
The final **model** and **tokenizer** will be pushed to an **S3 bucket** for storage and deployment.

![Artifact Upload Code](path/to/screenshot15.png)



### 15. **End**
The fine-tuning pipeline concludes here

---

## Ok, so now let's Talk about the Deployment/Inference Pipeline  🚀

*This is the diagram, of how the pipeline will look:*

![Deployment Pipeline](https://github.com/user-attachments/assets/83920759-ebe0-445b-9654-3fb86b3e6680)



## Deployment/Inference Pipeline 💥

### 1. Start  
A new commit is pushed to the main branch, triggering the **Continuous Integration (CI)** process.

---

### 2. Continuous Integration  
The **GitHub Actions Self-Hosted Runner** listens for changes in the main branch. It prepares the build process, and starts the Docker image build process.

---

### 3. Continuous Delivery  
- The **self-hosted runner** builds the Docker image using the `deployment` folder (which includes `app.py` and other required files).  
- The built Docker image is pushed to the **ECR (Elastic Container Registry)** for storage.  
- This completes the Continuous Delivery (CD) step.  

---

### 4. Continuous Deployment  
- On EC2, the **Flask endpoint** will pull the Docker image from **ECR** automatically using deployment scripts.  
- The Flask app is then run inside the container to expose the endpoint.

---

### 5. Input Prompt  
When a user enters a prompt via the Flask Web UI:  
- The system first checks if the model and tokenizer are already loaded in the EC2 instance's memory.  
- If not, it fetches them from the **S3 bucket** and loads them into memory.  
- The input is processed and passed to the model for inference.

---

### 6. End  
The output is generated and returned to the user via the Flask endpoint.

---



---




## **License 📜✨**  
This project is licensed under the MIT License.  
You are free to use, modify, and share this project, as long as proper credit is given to the original contributors.  
For more details, check the LICENSE file. 🏛️
