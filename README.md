# Transformer From Scratch (PyTorch)

<img width="689" height="761" alt="image" src="https://github.com/user-attachments/assets/9b99911a-5ec8-4da8-a265-17546dd9c0d5" />


A **from-scratch implementation of the Transformer architecture** based on *“Attention Is All You Need”* — built to deeply understand the internals of modern large language models rather than relying on high-level abstractions.

This project focuses on **correct data flow, tensor shapes, masking logic, and training dynamics**, and includes training a **decoder-only Transformer** on the Tiny Shakespeare dataset for character-level language modeling.

---

## 🔍 What This Project Covers

- Full Transformer implementation **without using `nn.Transformer`**
- Manual construction of:
  - Multi-Head Self Attention
  - Scaled Dot-Product Attention
  - Positional Encoding
  - Layer Normalization & Residual Connections
  - Feed Forward Networks
- Decoder-only architecture (GPT-style) as well
- Teacher forcing & causal masking
- Character-level language modeling
- Autoregressive text generation

---

## 🧠 Architecture Overview

### Implemented Modules
- **Embedding Layer**
- **Sinusoidal Positional Encoding**
- **Multi-Head Attention**
  - Explicit Q, K, V projections
  - Head splitting and merging
  - Scaled dot-product attention
- **Padding Masking**
- **Causal Masking**
- **Feed Forward Network**
- **LayerNorm + Residual Connections**
- **Stacked Decoder Blocks**
- **Linear Projection to Vocabulary**

> Every tensor reshape, transpose, and stride is handled explicitly to ensure conceptual clarity.

---

## 🔐 Masking Strategy

- **Causal (Look-Ahead) Mask**
  - Prevents tokens from attending to future positions
- **Padding Mask (where applicable)**

Masking is applied directly to attention score matrices *before softmax*, following the original paper.

---

## 📚 Dataset

- **Tiny Shakespeare**
- Character-level tokenization
- Vocabulary built from unique characters
- Input-output pairs created using sequence shifting

```text
Input  : "To be or not to"
Target : "o be or not to "
```
## Traning Details

- The model is trained as per the original paper
- Optimizer Used - Adam with Lambda Learning Rate Scheduler
- The Equation for learning rate scheduler is same as mentioned in paper
- <img width="660" height="65" alt="image" src="https://github.com/user-attachments/assets/d43b8758-1413-4ebf-a5a5-ee938092212a" />
- This equation is implemented in TinyShakespeare.ipynb

## Results

- After 5 epochs the training loss decreased to 0.6 ~ 0.4
- Eg of Generated Text is
  ```text
  context : ROM
  generated :
    ROMEO:
      You speak a thousand cut in my some pitched blood,
      Or seven faced by the silken sea
      For which they have placed their men of war?
  ```

