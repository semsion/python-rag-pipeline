import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, pipeline, GPT2LMHeadModel

# Initialize the sentence-transformers model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact and fast for embeddings

# Sample corpus for retrieval
corpus = [
    "Python is a popular programming language.",
    "AI stands for Artificial Intelligence.",
    "RAG models combine retrieval and generation for better context."
]

# Embed the corpus
corpus_embeddings = embedding_model.encode(corpus)

# Function to retrieve the most relevant document
def retrieve(query, corpus, corpus_embeddings, threshold=0.5):
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten()
    best_match_idx = np.argmax(similarities)
    if similarities[best_match_idx] < threshold:
        return "No relevant context found."
    return corpus[best_match_idx]

# Clean the output to normalize spaces and punctuation
def clean_output(text):
    # Find and keep only content after first "Answer:"
    answer_parts = text.split("Answer:", 1)
    if len(answer_parts) > 1:
        text = answer_parts[1]
    
    # Remove any subsequent Question/Answer patterns
    text = re.sub(r'Question:.*?Answer:', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'Question:.*?$', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'Answer:', '', text, flags=re.IGNORECASE)
    
    # Clean up spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def truncate_at_sentence_end(text, max_tokens):
    # Split into sentences more accurately
    sentences = re.split(r'([.!?])\s+', text)
    
    # Rejoin sentences with their punctuation
    complete_sentences = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            complete_sentences.append(sentences[i] + sentences[i+1])
    
    truncated_text = ''
    token_count = 0
    
    for sentence in complete_sentences:
        sentence_tokens = len(sentence.split())
        if token_count + sentence_tokens <= max_tokens:
            truncated_text += sentence + ' '
            token_count += sentence_tokens
        else:
            break
            
    return truncated_text.strip()

# Full RAG pipeline
def lightweight_rag(query):
    # Define the maximum number of tokens to generate
    max_new_tokens = 150

    # Retrieve the most relevant context
    context = retrieve(query, corpus, corpus_embeddings)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')

    # Generate a response using GPT-2 Large
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=50256
    )
    
    try:
        # Generate response
        response = generator(
            f"Context: {context}\nQuestion: {query}\nAnswer:", 
            max_new_tokens=max_new_tokens,
            truncation=True,
            num_return_sequences=1,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=50256
        )
        
        # Get the generated text
        generated_text = response[0]['generated_text']
        
        # Extract the answer part
        answer_start = generated_text.find("Answer:") + len("Answer:")
        if answer_start < len("Answer:"):
            return "Could not generate a valid answer."
        
        answer = generated_text[answer_start:].strip()

        # Basic cleaning
        cleaned_answer = clean_output(answer)
        
        # Truncate at the last complete sentence
        truncated_answer = truncate_at_sentence_end(cleaned_answer, max_new_tokens)

        print("Question:", query)
        print("Context:", context)
        print("Answer:", truncated_answer)
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "An error occurred while generating the response."

# Test query
query = "What is AI?"
lightweight_rag(query)