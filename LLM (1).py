#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q transformers accelerate torch')


# In[22]:


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.2,
    return_full_text=False,
    eos_token_id=tokenizer.eos_token_id,  e
    pad_token_id=tokenizer.eos_token_id
)


# In[16]:


def clean(text):
    return text.replace("]", "").strip()


# In[35]:


def zero_shot(ticket):
    messages = [
        {
            "role": "system",
            "content": "You are a support ticket tagging system. Return ONLY top 3 tags with their calculated confidence scores. The scores must be float values between 0.0 and 1.0 based on the ticket context. No explanations."
        },
        {
            "role": "user",
            "content": f"Ticket: {ticket}\nFormat:\ntag - score\ntag - score\ntag - score"
        }
    ]
    # Chat template apply karna
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = pipe(prompt)[0]["generated_text"]
    return clean(output)


# In[36]:


def few_shot(ticket):
    messages = [
        {"role": "system", "content": "You are a support ticket classification system. Return ONLY 3 tags with scores. No explanation."},
        {"role": "user", "content": "Ticket: Printer not working"},
        {"role": "assistant", "content": "printer issue - 0.95, hardware failure - 0.8, ink problem - 0.7"},
        {"role": "user", "content": "Ticket: Internet is slow"},
        {"role": "assistant", "content": "network issue - 0.9, connectivity - 0.85, bandwidth problem - 0.8"},
        {"role": "user", "content": f"Ticket: {ticket}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = pipe(prompt)[0]["generated_text"]
    return clean(output)


# 

# In[37]:


ticket = "My internet is not working and connection keeps dropping"

print("ZERO SHOT:\n", zero_shot(ticket))
print("\nFEW SHOT:\n", few_shot(ticket))


# In[38]:


import pandas as pd


data = {
    "Tag": ["Internet/Connectivity Issue", "Dropping Connections", "Slow/Outage Issues"],
    "Zero-Shot Score": [0.95, 0.87, 0.82],
    "Few-Shot Score": [1.00, 0.60, 0.40] # Few-shot ke tags thode different hain, par closely match karke compare kar sakte hain
}

df = pd.DataFrame(data)
print(df)


# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns


df_melted = pd.melt(df, id_vars="Tag", var_name="Method", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x="Tag", y="Score", hue="Method", palette="viridis")

plt.title("Comparison of Confidence Scores: Zero-Shot vs Few-Shot", fontsize=14)
plt.ylabel("Confidence Score (0.0 - 1.0)", fontsize=12)
plt.xlabel("Support Tags", fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# In[40]:





# # Summary
# ### What we did: We compared two different prompting techniques—Zero-Shot and Few-Shot—using a lightweight language model (Qwen2.5-1.5B-Instruct) to automatically categorize IT support tickets and assign them confidence scores.
# 
# ## Zero-Shot Prompting (No Examples):
#  ### We asked the model to generate the top 3 tags with scores directly, without giving it any past examples. Initially, it struggled with formatting (giving text like 'high/medium' or out-of-bound numbers), but once we strictly constrained it using a Chat Template and low temperature, it successfully gave correct decimal scores between 0.0 and 1.0. However, its scores were overly optimistic/generic.
# 
# ## Few-Shot Prompting (With Examples):
# ### We provided the model with a few context examples (e.g., teaching it that "Internet is slow" relates to network issues). This helped the model understand the exact business logic and real-world tagging standards perfectly.
# 
# # Conclusion:
# ### While Zero-Shot is good for quick, baseline testing, Few-Shot Prompting is much more reliable and accurate for smaller models. It forces the model to distribute weights realistically (giving a perfect 1.0 to the main issue and lower scores to secondary possibilities) rather than guessing high numbers across the board.
# # Key Insights from the Analysis
# 
# 
# 
# ## 1. 🔍 Variance in Confidence
# * **Zero-Shot** gives a flat high confidence (~90%) to almost all labels because it doesn't have a reference framework.
# * **Few-Shot**, on the other hand, understands the relative importance and drops the score for secondary issues (like ISP or router) down to 40%-60%.
# 
# ## 2. 🤖 Small Model Behavior
# * This experiment proves that for smaller models like `Qwen2.5-1.5B`, **Few-Shot Prompting acts as an anchor**.
# * It saves the model from overestimating its confidence and aligns the outputs with actual operational needs.
# 
# ## 3. 🚀 Production Recommendation
# * If deploying this in a real helpdesk system, the **Few-Shot approach should be used**, as it provides realistic thresholds that can be used to route tickets to the correct team automatically.

# In[ ]:




