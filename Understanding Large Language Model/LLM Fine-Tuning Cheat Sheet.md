Fine-tuning an LLM enables models to follow specific formats, perform specialized tasks, or operate consistently within a domain. However, it is not always the optimal solution. In some cases, a better prompt or a retrieval-augmented generation (RAG) pipeline can address the problem with less effort and computing resources. This cheat sheet outlines full fine-tuning, LoRA, QLoRA, other parameter-efficient fine-tuning (PEFT) methods, dataset preparation, training, evaluation, and key trade-offs.

All topics can be explored using open-source tools and locally available models. No paid APIs or platforms are required.

# LLM Fine-Tuning Cheat Sheet
## 1. Fine-tuning methods compared

Memory requirements vary based on model size, sequence length, batch size, precision, optimizer, and hardware. The following are relative comparisons, not fixed hardware guarantees.

Fine-tuning methods compared
Full fine-tuning is flexible but requires optimizer states, gradients, and model weights to fit within your compute budget. LoRA freezes the base model and trains additional low-rank matrices. QLoRA combines quantization with LoRA to further reduce memory requirements.

![](https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/Understanding%20Large%20Language%20Model/finetuning_methods_table.webp)

For most users working with a single GPU, LoRA or QLoRA is the most practical starting point.

## 2. When should you fine-tune instead of prompting or using RAG?
Select the technique that best addresses your specific problem.

![](https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/Understanding%20Large%20Language%20Model/technique_comparison_table.webp)
For example, if you require an assistant to answer questions about recently updated company policies, RAG is typically a better starting point than fine-tuning. If you need consistent structured outputs across many examples, fine-tuning may be appropriate.

These approaches can be combined. Fine-tune a model for a specific task, then use RAG to supply up-to-date information

## 3. Prepare your dataset before training
The effectiveness of fine-tuning depends largely on the quality of your examples.

For supervised fine-tuning (SFT), prepare examples containing an instruction and a desirable response. Depending on the task, you might use question-answer pairs, conversational messages, or prompt-completion records.

```python

from datasets import Dataset

examples = [
    {
        "text": "### Instruction:\nExplain overfitting.\n"
                "### Response:\nOverfitting occurs when a model "
                "fits training data too closely and generalizes poorly."
    },
    {
        "text": "### Instruction:\nDefine precision.\n"
                "### Response:\nPrecision is the fraction of "
                "positive predictions that are correct."
    }
]

dataset = Dataset.from_list(examples)

```python

Before training, remove duplicates, correct inaccurate answers, standardize formatting, and check for sensitive information. Divide your data into training, validation, and test sets, ensuring that near-duplicate examples do not appear in multiple splits.

For conversational datasets, use the chat template expected by the model. Hugging Face TRL supports standard text, prompt-completion, and conversational formats.

##  Fine-tune an LLM with LoRA using Hugging Face
Use Hugging Face Transformers, PEFT, and TRL to conduct a small fine-tuning experiment with open-source tools.

Install the dependencies:
```python

pip install torch transformers datasets peft trl accelerate
```python

The following example uses a small instruction-tuned model and LoRA. It serves as a starting point for experimentation and does not guarantee compatibility with all hardware.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

dataset = Dataset.from_list([
{"text": "### Instruction:\nExplain RAG.\n"
"### Response:\nRAG retrieves relevant external "
"information and supplies it to an LLM."},
{"text": "### Instruction:\nExplain LoRA.\n"
"### Response:\nLoRA adapts a model by training "
"small low-rank matrices while freezing base weights."}
])

config = LoraConfig(
r=8,
lora_alpha=16,
lora_dropout=0.05,
task_type="CAUSAL_LM"
)

training_args = SFTConfig(
output_dir="./lora-output",
num_train_epochs=2,
per_device_train_batch_size=1,
gradient_accumulation_steps=4,
learning_rate=2e-4,
logging_steps=5,
report_to="none"
)

trainer = SFTTrainer(
model=model,
args=training_args,
train_dataset=dataset,
peft_config=config,
processing_class=tokenizer
)

trainer.train()
trainer.save_model("./lora-adapter")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

dataset = Dataset.from_list([
    {"text": "### Instruction:\nExplain RAG.\n"
             "### Response:\nRAG retrieves relevant external "
             "information and supplies it to an LLM."},
    {"text": "### Instruction:\nExplain LoRA.\n"
             "### Response:\nLoRA adapts a model by training "
             "small low-rank matrices while freezing base weights."}
])

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    output_dir="./lora-output",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=config,
    processing_class=tokenizer
)

trainer.train()
trainer.save_model("./lora-adapter")

```python

This example demonstrates the core workflow. For actual experiments, use a sufficiently large, high-quality dataset and ensure the formatting matches the model’s training requirements. PEFT allows you to save and share the trained adapter without saving a full copy of the base model.


## 5. Understand the hyperparameters that matter

![](https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/Understanding%20Large%20Language%20Model/hyperparameters_table.webp)

Begin with conservative settings, monitor training and validation loss, and adjust only one or two parameters at a time. Higher LoRA ranks increase adaptation capacity but also raise the number of trainable parameters and memory requirements.

## 6. Evaluate the fine-tuned model
Do not evaluate a fine-tuned model solely based on its training loss.

Evaluate the fine-tuned model against the original model using the same held-out test cases. Select metrics that reflect the specific task:

Classification: Accuracy, precision, recall, and F1-score.
Structured outputs: Schema validity and field-level correctness.
Question answering: Answer correctness, relevance, and faithfulness.
Generation: Human review, task-specific checks, and appropriate text similarity metrics.

Additionally, test whether the model has lost general capabilities or acquired incorrect patterns from the training data.

The Takeaway
I recommend starting with prompting, establishing a reliable evaluation baseline, and then testing RAG or fine-tuning based on the application’s requirements.

Use LoRA for an accessible initial experiment, consider QLoRA when memory is a primary constraint, and pursue full fine-tuning only when its benefits outweigh the increased resource requirements.

The objective is not to fine-tune every model, but to identify the most suitable adaptation method, measure its impact on your application, and reliably reproduce results.

# References
[1](https://amanxai.com/2026/10/25/llm-fine-tuning-cheat-sheet/ fbclid=IwY2xjawVK7FJleHRuA2FlbQIxMQBwZG9mAXNydGMGYXBwX2lkEDIyMjAzOTE3ODgyMDA4OTIAAR58HbiicWLKpIKegxKMl0h6IysBMYd2pZUk18spaHiCB5xUFVpWgaY4InbGNA_aem_exmMvDg4RR5LLRuUUIbG-w)
