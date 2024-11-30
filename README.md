# 🎋🌿 **Production-Ready Instruction Fine-Tuning of Meta LLaMA 3.2 3B Instruct Project** 🌿🎉  
updating soon: 

## **Problem Statement**  
---  
*Note: This project simulates an industry-standard scenario where I am assuming the role of a developer at XYZ Company. The LLaMA 3.2 (3B) model has been successfully deployed in production as part of our product. However, to better serve our large user base of Kannada speakers, fine-tuning the model on a Kannada-specific dataset has become essential.*  




**High-Level Note:** All detailed information about **Instruction Fine-Tuning**, concepts like **reward models**, **training of large language models**, **Reinforcement Learning from Human Feedback (RLHF)**, **Direct Preference Optimization**, **Parameter-Efficient Fine-Tuning**, **Higher Floating-Point Precision Conversion to Lower Precision**, **Quantization**, **LoRA**, and **QLoRA** have already been covered in the `docs/` folder of this repository. Please refer to the documentation there for an in-depth understanding of how these concepts work. In this readme, I will go with project pipeline etc or the main details relevant to the project.


---  


# 🌟 **Fine-Tuning LLaMA 3.2 3B for Kannada Language Adaptation** 🌟  

At **XYZ Company**, we adopted the **LLaMA 3.2 (3B) model** as the core **AI foundation for our product** to provide cutting-edge AI-driven solutions. However, due to our **large Kannada-speaking user base**, the model required fine-tuning to better cater to their needs. After analyzing its performance, our **manager decided** that fine-tuning on a Kannada-specific dataset was essential to enhance the model’s effectiveness.


To achieve this, we are leveraging the **Hugging Face dataset** `charanhu/kannada-instruct-dataset-390k`, containing **390,000 high-quality rows of Kannada instructions**. This dataset serves as the foundation for fine-tuning the model to:  
-  **Better understand Kannada**: Improve comprehension of the language’s syntax, semantics, and nuances.  
-  **Generate accurate responses**: Ensure the model aligns with Kannada-speaking users' expectations and use cases.  
-  **Enhance the overall user experience**: Build a model that feels intuitive and responsive to Kannada-related queries.  




### **My Role as a Developer** 🎋  

As a developer, I am responsible for delivering a fine-tuned **LLaMA 3.2 3B** model that aligns with the defined **Key Performance Indicator (KPI)** objectives and ensures exceptional performance for Kannada-speaking users.  

- I will **instruct fine-tune** the model using the high-quality **Kannada dataset** from **Hugging Face** (`charanhu/kannada-instruct-dataset-390k`).  

- To address the constraints of **limited GPU resources**, I will implement **QLoRA-based 4-bit precision quantization** using **BitsAndBytes**, which involves:  
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

The **instruct-based fine-tuning** process will adhere to industry standards, ensuring the model is rigorously tested before being deployed for inference.  

### **Steps in My Approach**  

1. **Dataset Preparation**  
   - Use the **Hugging Face Kannada dataset** (`charanhu/kannada-instruct-dataset-390k`) for training, focusing on enhancing the model's performance in understanding and responding to Kannada-specific queries.  

2. **Efficient Training with Quantization**  
   - Optimize the training process by implementing **4-bit precision quantization** using **QLoRA** for efficient resource utilization.  
   - Leverage **Google Colab's limited GPU resources** to achieve faster training without compromising the quality of the fine-tuning process.  

3. **Model Deployment and Feedback Loop**  
   - Once the model is fine-tuned, it will be uploaded to an **S3 bucket** for easy access and deployment.  

This approach ensures a resource-efficient, scalable, and production-ready model tailored to meet the needs of Kannada-speaking users.  


**Note: This is high-level view.**


![CI_CD Diagram (3)](https://github.com/user-attachments/assets/b2aefea2-a17e-4242-9225-8701f5bbed3c)




--- 

## **Challenges Encountered** 🎋  

The project faced several challenges, including:  

- **Limited GPU Resources**: Fine-tuning a large model was challenging due to the scarcity of available GPU resources.  
- **Timeline Constraints**: A tight project timeline, driven by the large user base, required rapid action and attention.  
- **Model Inference on AWS**: Running inference on AWS incurred high costs, raising concerns around both **storage** and **compute expenses**.  



## **How I Fixed Challenges** 🌟  

- To address **GPU limitations**, I utilized **Google Colab** with **4-bit precision quantization** to enable efficient fine-tuning within the available resource constraints.  

- I worked closely with **prompt engineers** to accelerate the fine-tuning process, ensuring we met the project deadline despite the tight timeline.  

- For inference, I optimized model deployment by implementing a **multi-stage Docker setup** using **Docker Compose**, which helped reduce image size and enhance efficiency, lowering operational costs.  


---

###  Project System Design (or pipeline) 🎋🌿
Remember: For this project **Pipeline** is going to be seprated in two different parts

### **1. Finetuning Pipeline**  
- The **finetuning process** will be executed only once for this project.  
- **Quantization using `unsloth`:** The model is quantized to 4-bit precision, optimizing it for faster and more efficient finetuning.  
- **Fine-tuning LoRA layers:** These are trained in 16-bit precision for better accuracy. After fine-tuning, the LoRA layers are merged back into the quantized model.  
- Once fine-tuning is complete, the **merged model** along with the tokenizer is uploaded to an **S3 bucket**. This provides a centralized storage location and ensures that the model and tokenizer are ready for deployment or future use.  

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


---

## Deployment/Inference Pipeline 💥
---


---

## Ok, so now let's Talk about the Deployment/Inference Pipeline  🚀

*This is the diagram, of how the pipeline will look:*

![Deployment Pipeline](https://github.com/user-attachments/assets/83920759-ebe0-445b-9654-3fb86b3e6680)



## Deployment/Inference Pipeline 💥

### 1. Start  
A new commit is pushed to the main branch, triggering the **Continuous Integration (CI)** process.



### 2. Continuous Integration  
The **GitHub Actions Self-Hosted Runner** listens for changes in the main branch. It prepares the build process, and starts the Docker image build process.



### 3. Continuous Delivery  
- The **self-hosted runner** builds the Docker image using the `deployment` folder (which includes `app.py` and other required files).  
- The built Docker image is pushed to the **ECR (Elastic Container Registry)** for storage.  
- This completes the Continuous Delivery (CD) step.  



### 4. Continuous Deployment  
- On EC2, the **Flask endpoint** will pull the Docker image from **ECR** automatically using deployment scripts.  
- The Flask app is then run inside the container to expose the endpoint.



### 5. Input Prompt  
When a user enters a prompt via the Flask Web UI:  
- The system first checks if the model and tokenizer are already loaded in the EC2 instance's memory.  
- If not, it fetches them from the **S3 bucket** and loads them into memory.  
- The input is processed and passed to the model for inference.



### 6. End  
The output is generated and returned to the user via the Flask endpoint.

---
## Guide for Developers 🌿🎇✨💚🎆🌱🎇✨💚🎆 (this will be updated, post project completion)

If you want to build on top of this project, here are a few recommendations:

### Deployment Pipeline 🌱
The deployment pipeline will remain the same as outlined in the project. You can follow the same steps to build, test, and deploy the model using Docker, ECR, and EC2.

### Scripts for Deployment 📂
All the necessary scripts for deployment are available in the `scripts` folder:

- `ecr_scripts.sh`: For managing Docker images and pushing them to AWS ECR.
- `ec2_scripts.sh`: For deploying and running the Docker container on EC2.

### Fine-tuning 🌿
I recommend performing fine-tuning in Google Colab notebooks to save on computational resources. Colab provides a free GPU, making it an excellent environment for model fine-tuning.

### Environment Setup 🌱
To set up your environment for development or deployment:

1. **Create Conda Environment**:
    - Run the following command to create a Conda environment:
    ```bash
    conda create -n <env_name> python=3.10
    ```

2. **Activate the Environment**:
    - Activate the newly created environment:
    ```bash
    conda activate <env_name>
    ```

3. **Install Dependencies**:
    - Install the necessary dependencies listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4. **Environment Variables**:
    - Set up environment variables for AWS and other credentials. You can use `.env` files or export them directly in your terminal:
    ```bash
    export AWS_ACCESS_KEY_ID=<your_access_key>
    export AWS_SECRET_ACCESS_KEY=<your_secret_key>
    ```

### Tree Structure 🌱
To keep the README concise and organized, the detailed project tree structure is stored in a separate file: [tree_structure.md](tree_structure.md). This helps prevent the README from becoming too long and difficult to navigate.



Happy coding and deploying! 🎉💚

🎆💚✨🎉🎇💚🎆✨

---

## **License 📜✨**  
This project is licensed under the MIT License.  
You are free to use, modify, and share this project, as long as proper credit is given to the original contributors.  
For more details, check the LICENSE file. 🏛️
