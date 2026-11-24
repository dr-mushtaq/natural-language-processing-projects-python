I have spent the past months learning LLM inference, here are some of the best resources I have come across. Save this for your learning!
 1. 𝐅𝐨𝐮𝐧𝐝𝐚𝐭𝐢𝐨𝐧𝐬 𝐨𝐟 𝐋𝐋𝐌 𝐈𝐧𝐟𝐞𝐫𝐞𝐧𝐜𝐞
Goal: Understand what happens when you make an LLM call.
→ Tokenization, Embedding, Forward pass
→ Autoregressive generation
→ Prefill
→ Decode
→ KV Cache
→ TTFT and ITL 
→ Throughput and Latency

📌 𝐑𝐞𝐬𝐨𝐮𝐫𝐜𝐞𝐬
→ https://lnkd.in/exYXGE2T NVIDIA's talk at AI Engineering
→ https://lnkd.in/dTKWqm5g Huggingface LLM Inference
→ https://lnkd.in/dA4XJvSf great blog post by Abi Aryan

2. 𝐁𝐚𝐬𝐢𝐜 𝐓𝐫𝐚𝐧𝐬𝐟𝐨𝐫𝐦𝐞𝐫 𝐂𝐨𝐧𝐜𝐞𝐩𝐭𝐬 𝐟𝐨𝐫 𝐈𝐧𝐟𝐞𝐫𝐞𝐧𝐜𝐞
Goal: Understand the computation during inference. Focus only on the parts that matter for inference. 
→ Transformer block
→ Embeddings
→ Self-attention
→ Q, K, V

📌 𝐑𝐞𝐬𝐨𝐮𝐫𝐜𝐞𝐬
→ https://bbycroft.net/llm LLM visualizer

3. 𝐆𝐏𝐔 & 𝐇𝐚𝐫𝐝𝐰𝐚𝐫𝐞 𝐅𝐮𝐧𝐝𝐚𝐦𝐞𝐧𝐭𝐚𝐥𝐬
Goal: Understand the hardware constraints behind inference performance.
→ GPU architecture
→ Streaming Multiprocessors (SMs)
→ HBM vs SRAM
→ Memory hierarchy
→ Memory bandwidth
→ FLOPS
→ Compute-bound vs memory-bound workloads

📌 𝐑𝐞𝐬𝐨𝐮𝐫𝐜𝐞𝐬
→ https://lnkd.in/dUvixZy9 Make Deep Learning Go Brrr by 𝐇𝐨𝐫𝐚𝐜𝐞 𝐇𝐞 (A 𝐦𝐮𝐬𝐭 𝐫𝐞𝐚𝐝 to understand the bottleneck: Overhead, Memory Bandwidth, and Compute.
→ https://lnkd.in/dJpvP8Xg Basic Facts about GPUs

4. 𝐈𝐧𝐟𝐞𝐫𝐞𝐧𝐜𝐞 𝐎𝐩𝐭𝐢𝐦𝐢𝐬𝐚𝐭𝐢𝐨𝐧 𝐓𝐞𝐜𝐡𝐧𝐢𝐪𝐮𝐞𝐬
Goal: Understand how modern inference systems become fast by using inference optimization.
→ Quantization
→ PagedAttention
→ KV Cache Quantization
→ FlashAttention
→ Chunked prefill
→ Speculative decoding
→ Prompt caching
→ Continuous batching

📌 𝐑𝐞𝐬𝐨𝐮𝐫𝐜𝐞𝐬
→ https://lnkd.in/dQzgFECE NVIDIA: Mastering LLM Inference
→ Paper of each technique for a deep dive

5. 𝐈𝐧𝐟𝐞𝐫𝐞𝐧𝐜𝐞 𝐄𝐧𝐠𝐢𝐧𝐞𝐬
Goal: See how inference engines work. Different engines excel at different workloads.
→ vLLM: best general-purpose, battle-tested, built for high-throughput serving
→ SGLang: shines when requests share context: multi-turn chat, agentic tool calls, RAG
→ llama.cpp go-to for CPU and edge deployment

📌 𝐑𝐞𝐬𝐨𝐮𝐫𝐜𝐞𝐬
→ https://lnkd.in/dU5RCPJM Inside vLLM by Aleksa Gordić, best single deep dive
→ https://lnkd.in/d2xysQuP Distributed LLM Inference

Here is how I get started:
→ Learn the basic understanding of transformer and LLM's internal inference pipeline (prefill, decode)
→ Understand GPU hardware.
→ Understand key inference optimization techniques 
→ Explore an inference engine

If you are mid/senior ML/AI Engineer and want to go deep, study parallelism and distributed inference.

What else would you add to this list?

![resources](https://www.linkedin.com/posts/lanchuhuong_i-have-spent-the-past-months-learning-llm-share-7530646733941305345-WX0x/?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAAALfOjABJ0WirUOPXjd9IzMLuimcgHRZjPo)
