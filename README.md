# Self-Attention Implementation (ML)

A practical implementation of the self-attention mechanism, a fundamental building block of transformer architectures in modern machine learning.

## 📖 About

Self-attention is a key component in transformer architectures that allows models to weigh the importance of different parts of the input when processing sequences. This mechanism enables the model to focus on relevant information and capture long-range dependencies in data.

## 🎯 Project Overview

This project demonstrates the implementation of self-attention using a simple example sentence: **"Dream big and work for it"**

### Implementation Steps

#### 1. **Tokenization**
- The sentence is tokenized into individual words
- Each word is treated as a separate token
- Tokens: `["Dream", "big", "and", "work", "for", "it"]`

#### 2. **Attention Score Calculation**
- Attention scores are computed as the dot product of query and key vectors
- The dot product produces a scalar value representing the relevance between tokens
- **Example**: Using "big" as the query vector, we calculate attention scores against all key vectors (each token)
- This process determines how much attention the query token should pay to each key token

#### 3. **Normalization**
- Raw attention scores need to be normalized to sum to 1
- Two normalization approaches:
  - **Vanilla normalization**: Simple division by sum
  - **Softmax normalization**: Exponential normalization (used in this implementation)
- **Benefits**:
  - Improves interpretability of attention weights
  - Maintains LLM training stability
  - Prevents gradient vanishing/exploding issues

#### 4. **Context Vector Generation**
- Context vectors are computed by: **Input Embeddings × Attention Weights**
- The result is a weighted combination of input embeddings
- Each token gets a context-aware representation based on its relationship with other tokens

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Devikrishna545/Self-Attention-Implementation-ML-.git

# Navigate to the project directory
cd Self-Attention-Implementation-ML-

```
## 🔑 Key Concepts

- **Query (Q)**: The token we're currently focusing on
- **Key (K)**: All tokens that might be relevant to the query
- **Value (V)**: The actual information we want to extract
- **Attention Scores**: Dot product of Q and K, measuring relevance
- **Attention Weights**: Normalized attention scores (sum to 1)
- **Context Vector**: Weighted sum of values based on attention weights

## 📊 Mathematical Formulation

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

Where:
- `Q · K^T` computes attention scores
- `√d_k` is the scaling factor (dimension of key vectors)
- `softmax` normalizes the scores
- Multiplication with `V` produces context vectors

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

**Devikrishna545** - [@Devikrishna545](https://github.com/Devikrishna545)

Project Link: [https://github.com/Devikrishna545/Self-Attention-Implementation-ML-](https://github.com/Devikrishna545/Self-Attention-Implementation-ML-)

---

⭐ If you found this helpful, please consider giving it a star!
```
