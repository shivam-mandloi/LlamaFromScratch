# NeuroCpp-LlamaFromScratch

## 📌 Overview

This is a **learning-focused project** where I am implementing the **LLaMA-3 model from scratch in C++**.

The main goal of this project is **understanding**, not performance or production use.  
I want to deeply understand how a large language model works internally by **building everything manually**, step by step.

---

## 🎯 Project Goals

- Implement **LLaMA-3 architecture from scratch**
- Use **C++**, without any external ML or math libraries
- Only use **OpenBLAS (via WSL2)** for high-performance linear algebra operations
- Work with **raw arrays** for matrix and vector representation
- Focus on **CPU-based computation and optimization**
- First implement **forward propagation**
- Later try to implement **backward propagation (training)**
- Eventually explore **GPU support** (after CPU version is clear)

This project is meant to **build intuition**, not to create a highly optimized or production-ready LLM.

---

## 🧠 Learning Philosophy

- No shortcuts
- No black-box ML libraries (except OpenBLAS for optimized linear algebra)
- No Hugging Face APIs
- No pre-built deep learning frameworks in C++

Every component (attention, normalization, MLP, etc.) will be written manually so that the internal logic is fully clear.

---

## 🔗 Related Project (Reference)

This project is inspired by and follows ideas from my earlier work:

🔗 **NN-Blocks**  
https://github.com/shivam-mandloi/NN-Blocks

In **NN-Blocks**, I implemented neural network components from scratch in C++.  
This LLaMA project builds on the same philosophy and learning style.

---

## 🐍 Python → C++ Workflow

I started this project with a **Python implementation of the LLaMA model**, and this part is now **completed**.  
The Python version helped me clearly understand the model logic before moving to a low-level **C++ implementation**.

---

### ✅ Python Version (Completed)

- Implemented **LLaMA model from scratch**
- Implemented **Transformer, Attention, and other components manually**
- Uses **only PyTorch tensors**
- No Hugging Face model APIs are used for the implementation
- Can be used for **training or fine-tuning**
- Intended mainly for **learning and understanding**

#### 🔹 Weight Preparation (Important)

I used **Hugging Face only to download the pretrained weights**, not for model implementation.

To use the Python code:

1. First run **`LlamaWeightStore.ipynb`**
   - This notebook downloads LLaMA weights using Hugging Face
   - Saves all weights in a format used by this project

2. After saving the weights, run **`llama.ipynb`**
   - This notebook contains the **full LLaMA implementation**
   - Includes Transformer, Attention, and forward pass logic
   - Uses the saved weights for inference or experimentation

You **must run `LlamaWeightStore.ipynb` first**, otherwise the model will not work.

---

### 🚧 C++ Version (In Progress)

- C++ implementation will start now
- Built **fully from scratch**
- Uses **raw arrays** for vectors and matrices
- Uses **OpenBLAS (via WSL2)** for optimized linear algebra operations
- Focused mainly on **inference**
- Helps understand **low-level execution and memory handling**

The C++ version is the next step to deeply understand how LLaMA works at the system level.

---

This Python → C++ separation helps in learning both:
- **Model-level logic** (Python)
- **System-level implementation** (C++)


## ⚠️ Important Note

This is **not an optimized implementation** and **not intended for real-world deployment**.

- It is slow
- It is experimental
- It is incomplete

But it can be used as:
- A **base project**
- A **reference implementation**
- A **learning guide** for understanding LLaMA-style models

---

## 🚧 Current Status

- [x] Basic project setup
- [x] Python LLaMA implementation (from scratch)
- [ ] C++ forward pass (in progress)
- [ ] Autoregressive text generation
- [ ] Backward propagation
- [ ] CPU optimizations
- [ ] GPU support (future)

---

## 📌 Final Note

I hope to **complete this project before moving to the next one**.  

If you are also interested in learning **how LLMs work internally**, feel free to explore this project.