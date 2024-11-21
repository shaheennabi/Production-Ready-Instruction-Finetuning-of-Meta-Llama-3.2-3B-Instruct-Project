# Reinforcement Learning from Human Feedback with PPO



![Uploading Screenshot 2024-11-21 083539.png…]()


What is it, and why is it so confusing? Well, in this file, I will take you on a new adventure, and we will learn what **RLHF** with **PPO** actually means.

So, have you heard about the Reinforcement Learning field first? Oh! I guess that was the hot field in 2016, 2017, 2018... when Google DeepMind released the AlphaGo. Wow, it was amazing.  
Actually, the Reinforcement Learning field is itself a big field, but in this file, we will specifically talk about **PPO (Proximal Policy Optimization)**. It is an optimization algorithm. Let me take a step back to deep learning. You know how we train deep neural networks, right? It's the same as when we train deep neural networks to perform specific tasks, maybe generating text or classifying an image. Later, what we do to make our model robust is optimize it, as we have some of the best optimizers in the market, like Adam Optimizer, SGD, etc.  

The same is the story with **PPO**. It's an optimization algorithm in the field of Reinforcement Learning. Now you will see how we use **Reinforcement Learning** in this case. The algorithm from the **Reinforcement Learning family** is **Proximal Policy Optimization**, which is an optimization algorithm.

Let me talk about the process:

---

## Working of RLHF with PPO

As I mentioned in the previous file **reward_model.md**, we saw how the reward model was trained. The purpose of the reward model was to guide the main LLM (GPT model or ChatGPT) to generate responses that are more human-aligned.  

Now, let me talk about how we guide the model. For guiding the model, we use the **Proximal Policy Optimization algorithm**. So how does it work? When we prompt the large language model, like ChatGPT, it gives a response.

### Example 1:
**Prompt:** I like cricket  
**Response:** Yeah! Cricket was introduced in India by the British.

Now, do you observe how it answers? That is why we have to make it more human-specific.  

**After RLHF:**  
**Prompt:** I like cricket  
**Response:** Wow, cricket is a great game! Glad you like a sport.

Now see the difference between the responses.

---

When the main LLM generates the responses, the response is shown to the reward model. The reward model then gives the response a score, let's say 0.95. There are many examples like this, where the reward model is used to give a score for the response (i.e., how well the response aligns with humans).  

Later, what happens when the main LLM doesn't generate a good response as per human alignment? The main LLM is fine-tuned.  

How is it fine-tuned?  

We use **PPO** to fine-tune the model. The reward signal or score we get from the reward model is used to update the weights of the main LLM model.

---

### Reinforcement Learning (must have)

As you know, in Reinforcement Learning, we have an agent that interacts with the environment.  

In **PPO**, the reward signal guides the agent (here, the main LLM) to improve its policy (the way it generates responses). The loss is calculated using the reward signal and the advantage function from **PPO**.  

#### How does the update happen?
PPO updates the policy network (its architecture for generating responses) or fine-tunes it based on feedback.

---

### Over-Optimization and Frozen LLM

There is another thing to take care of, which is over-optimization.  

There is a concept of the **frozen LLM**—this is a copy of the original LLM. It's because fine-tuning the main LLM with the reward model might bias it too much toward the scores of the reward model.

- The reward model assigns high rewards to human-preferred responses.  
- By comparing the updated model's output with the frozen model, we ensure the updated model doesn't diverge too far from its original capabilities.

In short, we risk **over-optimization**.

---

### Example of Over-Optimization:
**Input:** What are the benefits of exercise?  
**Frozen LLM:** Exercise improves health and mood.  
**RLHF-fine-tuned (updated model):** Exercise is the ultimate key to living forever. Everyone should exercise daily.

While the updated model's response may score high on enthusiasm from the reward model, it deviates from realism. The frozen LLM helps curb such over-enthusiasm.

---

### KL Loss (Kullback-Leibler Divergence)

There is another concept called **KL Loss (Kullback-Leibler Divergence)**.  

KL divergence is a mathematical measure of how one probability distribution differs from another. In RLHF, KL loss compares:  
- The probability distribution of the updated model \(P(x)\).  
- The probability distribution of the frozen model \(Q(x)\).  

KL loss ensures the updated model doesn't overfit to the reward model's preferences.  
It penalizes large deviations, ensuring the LLM retains its general-purpose functionality.

--- 

This concludes the explanation of how RLHF with PPO works, focusing on frozen LLMs, over-optimization, and the role of KL loss.
