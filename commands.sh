# These commands set up a virtual environment and install the necessary dependencies to run the RAG pipeline.

# Create a virtual environment
python3 -m venv rag_env

# Activate the virtual environment
source rag_env/bin/activate

# Only run this command once initially to set up the environment
pip install transformers scikit-learn torch

# Run this command to run the RAG pipeline
python lightweight_rag.py