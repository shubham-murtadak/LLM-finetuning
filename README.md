# 🌟 **LLM-Finetuning with LLaMA 2 and QLoRA** 🌟

Welcome to the **LLM-Finetuning** repository! This project demonstrates how to fine-tune large language models like **LLaMA 2** and **Gamma Model** using parameter-efficient techniques such as **QLoRA** and **LoRA** for reduced computational overhead. 🚀

---

## 📝 **Project Overview**  
This repository focuses on fine-tuning models to handle specific tasks more effectively while using fewer resources. The **LLaMA 2** template is applied to instruction datasets to enhance performance with minimal VRAM usage.

### 🧠 **Models Fine-Tuned**  
- **LLaMA 2 (7B Chat Model)** 🤖  
  A powerful conversational model with impressive language generation capabilities.  
- **Gamma Model** 🌟  
  Designed to handle complex NLP tasks with high accuracy.  

---

## 📚 **Dataset**  

We reformat the **OpenAssistant Guanaco** dataset to follow the **LLaMA 2** template for better compatibility.

### 🔗 **Datasets Used**  
- **Original Dataset**: [OpenAssistant Guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)  
- **1K Sample Reformat Dataset**: [Guanaco-LLaMA2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)  
- **Complete Reformat Dataset**: [Guanaco-LLaMA2](https://huggingface.co/datasets/mlabonne/guanaco-llama2)  

To understand the dataset preparation, refer to this [Notebook](https://colab.research.google.com/drive/1Ad7a9zMmkxuXTOh1Z7-rNSICA4dybpM2?usp=sharing). 📓

---

## ⚙️ **Fine-Tuning Process**

### 🔧 **Parameter-Efficient Fine-Tuning (PEFT)**  
Since full fine-tuning requires extensive VRAM, we use **PEFT techniques** like **QLoRA** and **LoRA** to drastically reduce memory usage.

### 🛠️ **Steps**  
1. **Load Model:**  
   - Use `llama-2-7b-chat-hf` (chat version). 🦙  
2. **Dataset Training:**  
   - Fine-tune on [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k).  
   - Train for **one epoch** using **QLoRA** in **4-bit precision** with the NF4 type.  
3. **QLoRA Parameters:**  
   - **Rank**: 64  
   - **Scaling**: 16  

---

## 🛠️ **Installation & Setup**

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/shubham-murtadak/LLM-finetuning.git
   cd LLM-finetuning
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Fine-Tuning**  
   Open the respective Jupyter notebooks:
   - **LLaMA 3.2**: `llma_3.2_finetune/Fine_tune_Llama_2.ipynb`  
   - **Gamma Model**: `gamma_finetune/Fine_tune_Gamma.ipynb`  

---

## 🚀 **Fine-Tuning in Google Colab**  
💡 **Note:** Free Colab provides a **15GB GPU**. To work within these limits, we use QLoRA and 4-bit precision. Full fine-tuning isn't feasible, but PEFT techniques make it possible! 🎯  

---

## 💬 **Acknowledgments**  
- **Meta AI** for developing LLaMA.  
- **Hugging Face** for providing datasets and models.  
- **Community** for contributions to QLoRA & LoRA innovations.  

---

### 📧 **Contact**  
For any queries, feel free to reach out! 😊

--- 

Let’s make LLaMA 2 smarter together! 🦙
