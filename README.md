# python-rag-pipeline

This project implements a lightweight Retrieval-Augmented Generation (RAG) pipeline using Hugging Face Transformers (powered by PyTorch), scikit-learn, and Python. The pipeline:
- Retrieves relevant context from a corpus using semantic similarity
- Generates coherent responses using GPT-2 Large
- Ensures complete, well-formatted sentences
- Removes duplicate content and repeated phrases
- Handles errors gracefully

While the implementation includes various controls for response quality through parameter tuning and post-processing, as with any language model, outputs may occasionally diverge from the expected response or include hallucinated content.

The implementation focuses on both accuracy and response quality through careful parameter tuning and post-processing steps.

See below to get the project intialised, configured, and the dependencies installed.

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
    pip install transformers scikit-learn torch sentence-transformers

## Usage

Run the RAG pipeline with a sample query (the query is hard coded at the bottom of the file):
```sh
python lightweight_rag.py
```

You may receive an OpenSSL warning upon running the script. This can be safely ignored, or you can update to a different Python distribution that supports the required version of OpenSSL, recompile Python, or install an older version of urllib3.
