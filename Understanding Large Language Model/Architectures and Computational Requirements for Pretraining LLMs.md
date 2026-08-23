Welcome to another video on understanding large language models. Up until now, we have discussed how these models work, followed by videos on datasets and tokenization. Then we explored embeddings, and in the most recent video, we covered the attention mechanism. If you've watched all of these, you should now have a solid understanding of how an LLM processes and represents information at different stages.

In this video, we will focus on how these models are trained to generate text. We will break this down into three key aspects: model architecture, computational resources, and what it takes to train deep learning models of this scale.

Model Architecture and Parameter Counting
Let's start by looking at the architecture of GPT-2 Small. This model has:

An embedding dimension of 768
A context size of 1,024
12 decoder layers
12 attention heads
A total vocabulary size of 50,257
First, let's start by counting the total number of parameters in the model.

The embedding layer contains a learned embedding matrix which is trained during model training. This layer alone has approximately 38 million parameters. Next, we have positional encoding which helps the model understand word order. This can either be a fixed function or a trainable layer. In this case, we assume it is trainable, contributing about 0.8 million parameters.

Then comes the attention block, which consists of the query, key, and value matrices—all of which are trainable. Additionally, multi-head attention includes an output projection layer. Together, a single multi-head attention layer contains around 2.3 million parameters.

The feed-forward layer consists of approximately 4.7 million parameters. Here, we are ignoring bias and normalization parameters since they contribute only a few thousand parameters, which is negligible in comparison.

Repeating this structure across 12 layers results in approximately 85 million parameters. After that, we have the final linear layer. But since this layer shares parameters with the embedding layer, we do not count them separately. Summing up all these components, the GPT-2 Small model has a total of around 124 million parameters.

However, this is considered a small model compared to modern large-scale LLMs, which now contain billions or even trillions of parameters. These larger models have deeper architectures with more trainable layers, larger model dimensions, and additional repetitions, making them significantly more powerful.

Training Datasets and Training Steps
Now let's talk about the dataset. These models are trained on large datasets containing billions or even trillions of tokens. To prepare the training data, we take a fixed context size and batch size. For example, if we take a context size of 10 and a batch size of two, the training data and its labels are structured by shifting the tokens one step ahead for the labels. This process happens after tokenization, ensuring that each training example contains the exact number of tokens needed.

In traditional machine learning, you might have seen models trained for multiple epochs, meaning the entire dataset is passed through the model multiple times. This helps improve learning by allowing the model to see the same data repeatedly and adjust its parameters accordingly. However, when training large language models, we do not usually count epochs. Instead, we use what is called training steps.

This is because the datasets are massive, often in the range of billions or trillions of tokens, making it infeasible and unnecessary to repeat the same training examples multiple times. In most cases, each training example is used only once.

For example, if we have a total training dataset size of 50 million tokens and we set a context size of 1,024 with a batch size of 32, then each training step processes 32 sequences of 1,024 tokens. To compute the total number of training steps required for one full pass through the dataset, we divide 50 million tokens by the total tokens processed in one step ($1,024 \times 32$), which gives us approximately 1,526 training steps.

In the real world, pre-training on 50 million tokens is a very small dataset. Typically, pre-training involves datasets with trillions of tokens, meaning the total number of training steps would be in the thousands or even more. In research papers and reports, you will often see statements like "Model X trained on Y amount of tokens," which refers to the total number of tokens processed during training.

Memory Requirements for Training
Now let's talk about the memory requirements needed to train these models. Let's consider a model with 1 billion parameters using full-precision floating-point representation (32-bit) to store each parameter. Just to store the model weights, we would need around 4 GB of memory. However, this is only for loading the model weights, and actual training requires much more memory.

The first thing that needs to be stored is the model weights. In addition, each neuron produces an output activation when given an input, and these activations must be stored in memory. During backpropagation, gradients are computed and also need to be stored. If we are using an optimizer like Adam, which is commonly used in large language models, we also need to store the optimizer state in memory. This is crucial for efficient training.

The memory requirement for gradients is the same as the model parameters since each parameter has both a weight value and a gradient value. Additionally, Adam requires storing two extra values for each parameter, meaning we need double the size of the model weights to store the optimizer state.

On top of this, we need to account for activation memory. During training, each neuron produces output activations, and the total memory required depends on batch size, context size, and the number of layers in the model. More input tokens mean more activations. So roughly, activation memory can take around 10 to 15 GB in our case.

Many people may not consider activation memory, and even if everything looks fine during model loading, they might run into a CUDA out-of-memory error when training starts. Considering all these factors, training a 1 billion parameter model using full-precision floating-point requires around 30 GB of memory.

However, today's large language models contain hundreds of billions of parameters, which require significantly more computational resources. For training large models, high-performance GPUs are needed.

Gemini Notebook can be inaccurate; please double check its responses.
