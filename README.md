# python-rag-pipeline

This project implements a lightweight Retrieval-Augmented Generation (RAG) pipeline using Hugging Face Transformers, scikit-learn, python, and PyTorch. The pipeline retrieves relevant context from a corpus and generates responses based on the context and query.


See the steps below to get the project intialised, configured, and dependacies installed.

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv rag_env
    source rag_env/bin/activate
    ```

3. Install the necessary dependencies:
    ```sh
    pip install transformers scikit-learn torch

## Usage

Run the RAG pipeline with a sample query:
```sh
python lightweight_rag.py
```

