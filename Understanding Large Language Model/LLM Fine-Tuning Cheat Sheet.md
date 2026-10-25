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

# References
[1](https://amanxai.com/2026/10/25/llm-fine-tuning-cheat-sheet/?fbclid=IwY2xjawVK7FJleHRuA2FlbQIxMQBwZG9mAXNydGMGYXBwX2lkEDIyMjAzOTE3ODgyMDA4OTIAAR58HbiicWLKpIKegxKMl0h6IysBMYd2pZUk18spaHiCB5xUFVpWgaY4InbGNA_aem_exmMvDg4RR5LLRuUUIbG-w)
