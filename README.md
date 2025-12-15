# Efficient Sentiment Analysis with LoRA & DistilBERT

This project implements a lightweight sentiment analysis model using **Parameter-Efficient Fine-Tuning (PEFT)**. By leveraging **LoRA (Low-Rank Adaptation)**, it fine-tunes a pre-trained `distilbert-base-uncased` model on the IMDB dataset without updating all model parameters, significantly reducing memory usage and training time.

Unlike standard Hugging Face implementations, this project utilizes a **custom, manual PyTorch training loop** rather than the high-level `Trainer` API, offering granular control over the training process and optimization steps.

## Key Features

* **Model:** DistilBERT Base (Uncased).
* **Technique:** LoRA (Rank=4) for efficient adaptation.
* **Dataset:** Truncated IMDB (Binary Classification: Positive/Negative).
* **Architecture:** Custom PyTorch training loop with `AdamW` optimizer and dynamic padding.
* **Hardware Aware:** Auto-detects **CUDA** (NVIDIA), **MPS** (Mac Silicon), or **CPU**.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install jupyterlab ipykernel ipywidgets datasets torch torchvision torchaudio transformers peft evaluate scikit-learn tqdm
    ```

##  Project Structure

The core logic is contained within the Jupyter Notebook (or Python script):

1.  **Data Loading:** Fetches `shawhin/imdb-truncated` using the Hugging Face `datasets` library.
2.  **Preprocessing:** Tokenizes text with `distilbert-base-uncased` tokenizer and formats columns for PyTorch.
3.  **LoRA Configuration:** Injects trainable rank decomposition matrices into the query/linear layers (`q_lin`) while freezing the base model.
4.  **Manual Training Loop:**
    * Iterates through epochs using `DataLoader`.
    * Computes loss and backpropagates manually.
    * Evaluates accuracy at the end of every epoch.
5.  **Inference:** Loads the saved adapter and tests on unseen sentences.
