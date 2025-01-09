from transformers import AutoTokenizer, AutoModel, pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Sample corpus for retrieval
corpus = [
    "Python is a popular programming language.",
    "AI stands for Artificial Intelligence.",
    "RAG models combine retrieval and generation for better context."
]

# Embedding function using a BERT-like model
def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Embed entire corpus
def embed_corpus(corpus, tokenizer, model):
    return [embed_text(doc, tokenizer, model) for doc in corpus]

# Retrieve the most relevant document
def retrieve(query, corpus, corpus_embeddings, tokenizer, model):
    query_embedding = embed_text(query, tokenizer, model)
    similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten()
    best_match_idx = np.argmax(similarities)
    return corpus[best_match_idx]

# Full RAG pipeline
def lightweight_rag(query):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.config.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id

    # Embed corpus documents
    corpus_embeddings = np.vstack(embed_corpus(corpus, tokenizer, model))

    # Retrieve the most relevant context
    context = retrieve(query, corpus, corpus_embeddings, tokenizer, model)

    # Load a lightweight generator model for text generation
    generator = pipeline('text-generation', model='distilgpt2', tokenizer='distilgpt2')

    # Generate response with a higher max_length
    response = generator(
        f"Context: {context}\nQuestion: {query}\nAnswer:", max_length=200, do_sample=True, truncation=True
    )
    
    # Print the full generated text
    generated_text = response[0]['generated_text']
    print("Answer:", generated_text)

# Example usage
query = "What is python?"
lightweight_rag(query)