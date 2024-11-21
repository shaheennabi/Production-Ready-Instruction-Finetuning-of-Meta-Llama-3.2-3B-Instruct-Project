##  Reward Model Working 

![Reinforcement-Learning-1](https://github.com/user-attachments/assets/be55dc2c-02b4-4779-b5e4-7d5e0a00ca45)

So, before going and looking deep inside this project, I will explain some important things that how the **Large Language Models** are actually trained. In this file I will explain about the **reward model**.


As you can see in the image that we prompt a large language model like ChatGPT based on GPT4o or other models running in their backend, we get a response back. For Example, I am asking ChatGPT I like to play Cricket and gives the response back as (Response 1): Ok! It's good to know that you like Cricket.

Now what you observe here, let me tell you when ChatGPT was initially launched in Nov-Dec 2022, it was good at many things, sometimes it was misleading like giving some wrong answre as in the example, you can see, It says Ok! It's good to know that you like Cricket, what can you observe from my opinion this is not the best response I am expecting, the response should be like quote(Response 2): Wow! Cricket is a great game as it improves your health and you mental abitlity etc. Now can you see the differences of the reponses.

Response 2 is far better than Response1, so it was big thing for researchers to make sure llm's generate the responses that are more human preferred (i mean how we humans expect). There is the story of **Reward Model**, it is a different model than main llm in our case let it be any GPT model. So what reward model does is when the main llm (GPT model) generates the responses, it generates the score (aka reward signal). Let me describe it briefly

## Working of Reward Model

Remember reward model is a seperate model, that is trained seperately on Responses the main llm(GPT model) generates.

For example: 
let's ask ChatGPT 

prompt: Hi, how are you?

**Response1:** I am good?

and few more responses:

**Response2:** Hi, there I am doing great!

**Response3:** Thanks for asking, I am doing great, how have you been.

Now doo you observe something in these response can I say one thing that **Response3** is far better or more human aligned than **Response2** and **Response1**. Can I give it a ranking like this: **Response3**>**Response2**>**Response1**.

This is the same anotomy inside reward model, atually when the main llm(ChatGPT model) generates responses, these responses are collected and bought together and human labellers in this case **researchers** rank these responses that are best aligned with human prefereces(like we humans expect)

So as you can see in the figure, the responses from llm are labelled by human annotators and they rank, and for each **Response** the score is assigned like in our case:
**Response3** = 0.95
**Response2** = 0.7
**Response1** = 0.4

So, we want **Response3** like responses our main llm should generate. So what we do we combine these response I mean, prompt: Hi, how are you, Response: Thanks for asking, I am doing great, how have you been, Score = 0.95

So here we use prompt, response as **independent variable** for model, and **score** is dependent variable, I mean the target model has to predict.

Reward model is trained on the same architecture like the transformer based architecture, same like how we train deep neural networks, as you can see we give prompt, resposne in **input** and **score** as output. So what happens model is trained on it and it gives out the probability or score as **output**, when we give it a prompt, response.

Let me clear it briefly:

Same as we train any deep neural network **Reward Model** is trained the same and it has to provide the scalar output (**score**).

So what happens when the reward model is trained on these mentioned things, it is used for finetuning the main llm (i mean the ChatGPT or GPT model), **and I will tell you, how this **Reward Model** is used later to finetune or guide the main llm in the next file named as: RLHF with PPO**.



