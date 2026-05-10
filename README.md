You can use the model directly from HuggingFace: https://huggingface.co/Joaaaane


---
library_name: transformers
tags:
- text summarization
license: apache-2.0
language:
- en
metrics:
- rouge
pipeline_tag: text2text-generation
---

# Model Card for Post-Disaster Digital Help Desk Summarization Model

<!-- Provide a quick summary of what the model is/does. -->
This model is designed to summarize digital help desk conversations in post-disaster scenarios, specifically tailored for non-profit organizations providing aid. It is based on the BART model, fine-tuned using parameter-efficient methods like LoRa adapters.

## Model Details

### Model Description
This is a parameter efficient finetuned model based on the fine-tuning of the BART model. the methodology used is the LoRa adapter. this model focuses on automated text summarization of digital helpdesk conversations in post-disaster assistance scenarios in order to improve the efficiency and quality of the information gathered to provide timely and effective support to the affected people.
## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
The model is designed to summarize digital help desk conversations for nonprofit organizations in post-disaster assistance scenarios, helping digital help desk staff to quickly extract key information and reduce the time it takes to manually write high-quality summaries.

## Bias, Risks, and Limitations
Generated summaries may contain certain errors, such as the inclusion of sensitive information, and require manual secondary correction to ensure accuracy and privacy protection.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
# install package
!pip install transformers[torch] -U
!pip install -q -U peft

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import notebook_login

# login to hugging_face
notebook_login() # use model on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# load base model
model_name = "knkarthick/MEETING_SUMMARY"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load trained adapter
adapter_id = "Joaaaane/510_ABW_LoRaAdapter_PostDisasterConv"
model.load_adapter(adapter_id) # set the model to evaluation mode
model.eval()
input_text = """
PA: Hello, I need urgent housing help as a refugee from Ukraine. Can you assist?
agent: Hello, thank you for reaching out to the Red Cross. We’re here to help with housing.
agent: Have you registered with the local authorities yet?
PA: Yes, but they mentioned delays, and we need something soon. It's urgent.
agent: We have temporary shelters available. How many are with you, and are there any special needs?
PA: It's just me and my elderly mother; we need accessible housing.
agent: We can arrange for accessible temporary shelter. I’ll expedite your request and aim to place you within a few days.
agent: I'll also connect you with a Ukrainian-speaking volunteer to help with your paperwork and make your mother more comfortable.
PA: Thank you so much. This help means a lot to us right now.
agent: You're welcome! Expect a call from our volunteer by tomorrow. We’ll make sure you both are settled quickly.
PA: Thanks again. Looking forward to resolving this soon.
"""

# tokenized inputs
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
# generate summary tokens
outputs = model.generate(inputs['input_ids'], max_length=62, num_beams=5, early_stopping=True)
# decode tokens
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Summary:", summary)
```

## Training Details

### Training Data
Data provided by 510, an initiative of the Netherlands Red Cross (all confidential data has been masked).

### Testing Data
Data provided by 510, an initiative of the Netherlands Red Cross (all confidential data has been masked).

### Metrics
ROUGE Score

### Results

| Metric             | Before LoRA | After LoRA |
|--------------------|-------------|------------|
| **ROUGE 1**        | 22.50       | 28.30      |
| **ROUGE 2**        | 4.96        | 8.64       |
| **ROUGE L**        | 17.24       | 22.50      |

## Citation

Base model: https://huggingface.co/knkarthick/MEETING_SUMMARY
