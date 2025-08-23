1. Motivation & Problem

Before Transformers, sequence-to-sequence models (for tasks like machine translation) relied heavily on RNNs (LSTMs, GRUs) or CNNs.
These had two big issues:

Sequential bottleneck: RNNs process one token at a time, making parallelization hard.

Difficulty with long dependencies: Modeling relationships between far-apart words is inefficient.

The paper proposes a new architecture: The Transformer, which removes recurrence and convolution entirely, relying only on attention mechanisms.

2. Core Idea: Self-Attention

The model builds representations by relating all tokens to each other directly through self-attention.

Scaled Dot-Product Attention

Given a query Q, keys K, and values V:

𝐴
𝑡
𝑡
𝑒
𝑛
𝑡
𝑖
𝑜
𝑛
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
(
𝑄
𝐾
𝑇
𝑑
𝑘
)
𝑉
Attention(Q,K,V)=softmax(
d
k
	​

	​

QK
T
	​

)V

The scaling by 
𝑑
𝑘
d
k
	​

	​

 avoids large gradients for high-dimensional vectors (see page 4, Fig. 2).

Multi-Head Attention

Instead of one attention function, the Transformer projects queries/keys/values into multiple smaller subspaces (heads).

Each head learns different relationships (syntax, semantics, positional cues).

Outputs are concatenated and linearly projected.

(Figure 2 on page 4 shows both scaled dot-product attention and multi-head attention visually.)

3. Model Architecture

The Transformer follows an encoder-decoder design (Figure 1, page 3):

Encoder: 6 stacked layers, each with:

Multi-head self-attention

Feed-forward network

Residual connection + LayerNorm

Decoder: 6 stacked layers, each with:

Masked multi-head self-attention (prevents looking ahead)

Encoder-decoder attention

Feed-forward network + LayerNorm

Positional Encoding

Since no recurrence/convolution is used, position information is injected by adding sinusoidal encodings to embeddings (page 6).

4. Why Self-Attention?

Compared to RNNs and CNNs:

Complexity: Self-attention is 
𝑂
(
𝑛
2
⋅
𝑑
)
O(n
2
⋅d), but parallelizable.

Path length: Any two positions are connected in 1 step, unlike RNNs (O(n)) or CNNs (O(log n)).

Interpretability: Attention heads learn interpretable relations (see visualizations on pages 13–15).

5. Training

Datasets: WMT 2014 English–German (4.5M pairs), English–French (36M pairs).

Hardware: 8 × NVIDIA P100 GPUs.

Optimizer: Adam with warmup learning rate schedule (page 7).

Regularization: Dropout & label smoothing.

6. Results

English → German: Transformer (big) scored 28.4 BLEU, 2+ points better than previous best.

English → French: 41.8 BLEU, state-of-the-art at much lower training cost.

The model also generalized well to English constituency parsing (page 10).

The table on page 8 (Table 2) shows that the Transformer outperformed CNN- and RNN-based models while requiring fewer FLOPs.

7. Key Contributions

Introduced the Transformer architecture.

Showed that attention alone is sufficient for sequence modeling.

Demonstrated faster training and better performance than RNN/CNN models.

Laid the foundation for modern models like BERT, GPT, T5, etc.