1-[Transformers in NLP Explained: Architecture, Attention, and Why They Replaced RNNs](https://chatgpt.com/c/699a8271-4abc-8323-85df-f817da2a175e)




# Transformers in NLP: A Practical Overview (Without the Hype)

There’s been a lot of noise around transformers. And honestly, some of it is deserved.

Since their introduction in 2017, transformer models have become the foundation of modern natural language processing (NLP). Models like BERT, T5, and GPT are all built on this architecture.

But what actually makes transformers different?

Let’s break it down in plain language. Then we’ll connect it to a simple Python example so you can ground the ideas in code.

# The Origin of Transformers

The Transformer architecture was introduced in the 2017 paper “Attention Is All You Need” by researchers at Google.

That paper changed NLP research. Instead of relying on recurrent neural networks (RNNs), the authors proposed a model built entirely around attention mechanisms.

And that shift solved some long-standing problems.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/Week%202(Sumarization)/Pic1.png"></a>
</p>

There has been a lot of hype with the transformers. In this blog, I'll give you an overview of the transformers model. The transformer model was introduced in 2017 by researchers at Google, including Lukasz Kaiser, who helped us develop this course. Since then, the transformer architecture has become the standard for large language models, including BERT, T5, and GPT-3, which you'll learn about later. The transformers revolutionized the field of natural language processing. I suggest that you read the first transformer paper, Attention is all you need. It's the basis for all the models presented in the rest of this course. You'll see how each part of the transformer model works in detail. But first, I want to give you a brief
overview of this architecture. Now, don't worry if some of its components aren't clear, I'll go more in depth on the following lectures.

# Core Components of the Transformer Architecture

Let’s walk through the main building blocks of transformer models.

## 1. Self-Attention (Scaled Dot-Product Attention)

Self-attention allows each word in a sentence to “look at” every other word.

So instead of understanding words in isolation, the model builds contextual meaning.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/Week%202(Sumarization)/pic2.png"></a>
</p>

For example:

“The bank approved the loan.”

The word “bank” can mean different things. Self-attention helps the model use surrounding words like “loan” to understand the correct meaning.

And technically, it works using matrix multiplications — which makes it efficient and GPU-friendly.

## 2. Multi-Head Attention

Instead of computing attention once, transformers compute it multiple times in parallel.

Each “head” focuses on different relationships in the sentence.

Some heads might learn:

Grammar structure

Subject-object relationships

Long-distance dependencies

Then the results are combined.

This makes representations richer and more expressive.

## 3. Encoder and Decoder Structure

The original transformer architecture has two main parts:

### Encoder

- Multi-head self-attention

- Residual connections

- Layer normalization

- Feed-forward network

- Repeated multiple times

The encoder creates contextual embeddings for each input token.

### Decoder

- Masked self-attention (can’t see future tokens)
- Encoder-decoder attention
- Feed-forward layers
- Repeated multiple times
- 
The decoder generates output step-by-step (for example, in machine translation).

##  4. Positional Encoding

Transformers don’t use recurrence. So they don’t automatically understand word order.

To fix that, they add positional encoding to embeddings.

This gives each token information about its position in the sequence.

Without positional encoding, word order would be lost — which would break language understanding.


The Transformer model uses scale dot-product attention, The first form of attention is very efficient in terms of computation and memory due to it consisting of just matrix multiplication operations. This mechanism is the core of the model and it allows the transformer to grow larger and more complex while being faster and using less memory than other
comparable model architectures. 

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/Week%202(Sumarization)/pic3.png"></a>
</p>

In the transformer model, you will use the multi-head attention layer. This layer runs in parallel and it has a number of scale dot-product attention mechanisms and multiple linear transformations of the input queries, keys, and values. In this layer, the linear transformations are learnable parameters. 

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/Week%202(Sumarization)/pic4.png"></a>
</p>

The transformer encoder starts with a multi-head attention module that performed self attention on the input sequence. That is, each word in the input attends to every other word in the input. This is followed by a residual connection and normalization, a feed forward layer, and another residual connection and normalization. This entire block is one encoder layer and
is repeated N number of times. Thanks to self attention layer,
the encoder will give you a contextual representation
of each one of your inputs. The decoder is constructed similarly
to the encoder with multi-headed attention modules,
residual connections, and normalization. The first attention module is
masked such that each position attends only to previous positions. It blocks leftward flowing information. The second attention module
takes the encoder output and allows the decoder to attend to all items. This whole decoder layer is also repeated
some number of times, one after another. Transformers also incorporates
a positional encoding stage which encodes each input's position in the sequence. This is necessary because transformers
don't use recurrent neural networks, but the word order is relevant for
any language. Positional encoding can be learned or
fixed, just as with word embeddings. For instance, let's suppose you want
to translate from the French phrase. Over here you have [FOREIGN], and then you want to capture
the sequential information. The transformers uses a positional
encoding to retain the position of the input sequence. The positional encoding has values that
are added to the embeddings so that for every input word you have information
about its order and position. In this case, a positional encoding
vector for each word, [FOREIGN]. Putting these parts together,
here's the full model architecture. Briefly on the left,
the input sentence is first embedded and the positional encodings are applied. This goes to the encoder, which consists of multiple layers
of multi-head attention modules. On the right is the decoder,
which takes the output sentence, shifts it over one step to the right,
and the outputs from the encoder. The decoder output is turned
into output probabilities using a linear layer with a softmax activation. This architecture is easy to
parallelize compared to RNN models, and as such, can be trained much more
efficiently on multiple GPUs. It can also scale up to learn multiple
tasks on larger and larger datasets. I went through this quickly but
don't worry, I'll go in-depth on each
part in later videos. In summary, RNNs have some problems that
come from their sequential structure. With RNNs, it is hard to fully exploit
the advantages of parallel computing. And for long sequences, important
information might get lost within the network and
vanishing gradient problems arise. But fortunately, recent research
has found ways to solve for the shortcomings of RNNs
by using transformers. Transformers are a great alternative
to RNNs that help overcome these problems in NLP and in many fields
that process sequential data. You now can see why everyone
is talking about transformers, they are indeed very useful. In the next video, I'll talk about some
of the applications of transformers.










