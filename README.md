# NeuroCpp-LlamaFromScratch

## 📌 Overview

This is a **learning-focused project** where I am implementing the **LLaMA-3 model in C++**.

The main goal of this project is **deep understanding**, not performance or production use.  
I want to understand how a large language model works internally by **building core components manually**, step by step.

---

## 🎯 Project Goals

- Implement **LLaMA-3 architecture**
- Use **C++ with ATen (LibTorch backend)** for tensor operations
- Use **ATen only for optimized tensor computation**
  - Matrix multiplication
  - Tensor storage
  - Basic tensor operations
- Implement all core components manually:
  - Attention
  - Linear layers
  - Feedforward Network (FNN)
  - Normalization
- Focus on **CPU-based execution**
- Explore **GPU support (future)**

This project is meant to **build intuition at both model-level and system-level**.

---

## 🧠 Learning Philosophy

- No black-box deep learning frameworks
- No Hugging Face model APIs
- No pre-built Transformer implementations

Only **ATen is used as a low-level tensor engine**.

👉 Important idea:
- **ATen = fast tensor operations**
- **Everything else = written from scratch**

---

## 🔗 Related Project (Reference)

This project is inspired by my earlier work:

🔗 **NN-Blocks**  
https://github.com/shivam-mandloi/NN-Blocks

In **NN-Blocks**, I implemented neural network components from scratch in C++.  
This project builds on the same idea but moves toward **Transformer-scale models**.

---

## 🐍 Python → C++ Workflow

I first implemented the model in Python to clearly understand the architecture, and then translated that understanding into C++.

---

### ✅ Python Version (Completed)

- Full **LLaMA-style model implemented from scratch**
- Transformer, Attention, and other components written manually
- Uses **PyTorch tensors only (no model APIs)**
- Can be used for training or experimentation

#### 🔹 Weight Preparation

I used Hugging Face **only to download pretrained weights**, not for implementation.

Steps:

1. Run `LlamaWeightStore.ipynb`
   - Downloads and stores weights

2. Run `llama.ipynb`
   - Runs full model implementation using saved weights

---

### ✅ C++ Version (Completed - Core Inference)

- Implemented LLaMA using **C++ + ATen**
- **ATen is used only for:**
  - Tensor representation
  - Efficient matrix multiplication
  - Basic tensor operations

- **All model logic is implemented manually:**
  - Attention mechanism
  - KV cache
  - Linear layers
  - Feedforward network
  - Masking logic

- Implemented:
  - ✅ Forward pass
  - ✅ Autoregressive text generation
  - ✅ KV cache optimization

- Focus areas:
  - Understanding memory layout
  - Efficient tensor usage
  - Low-level Transformer execution

---

## ⚠️ Important Note

This is **not a production-ready implementation**.

- Not optimized for large-scale deployment
- Experimental and learning-focused

But useful as:
- A **learning guide**
- A **reference implementation**
- A **base for deeper research**

---

## 🚧 Current Status

- [x] Project setup
- [x] Python LLaMA implementation (from scratch)
- [x] C++ forward pass (ATen-based)
- [x] Autoregressive generation
- [x] KV cache optimization
- [ ] GPU support (future)

---

## 📌 Final Note

The goal is to **fully understand LLaMA from both perspectives**:

- **High-level (Python)** → model logic  
- **Low-level (C++)** → execution, memory, performance  

If you're also trying to understand LLMs deeply, this project may help you think beyond frameworks.