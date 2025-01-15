import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, pipeline

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

# Check if text is a properly formatted complete sentence
def is_complete_sentence(text):
    if not text:
        return False
    first_word = text.split()[0] if text.split() else ''
    has_capital = first_word and first_word[0].isupper()
    has_end_punct = text.rstrip()[-1] in '.!?'
    return has_capital and has_end_punct

# Full RAG pipeline
def lightweight_rag(query):
    # Retrieve the most relevant context
    context = retrieve(query, corpus, corpus_embeddings)

        # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

    # Generate a response using distilgpt2
    generator = pipeline(
        'text-generation',
        model='distilgpt2',
        tokenizer=tokenizer, 
        do_sample=True,
        temperature=0.4,
        top_p=0.92,
        top_k=40,
        max_length=100,
        pad_token_id=50256
    )
    
    try:
        # Generate response
        response = generator(
            f"Context: {context}\nQuestion: {query}\nGive a clear, direct answer:", 
            max_length=100,
            do_sample=True,
            truncation=True,
            num_return_sequences=1,
            repetition_penalty=1.2,
            length_penalty=1.0
        )
        
        # Get the full generated text
        generated_text = response[0]['generated_text']
        
        # Answer extraction
        answer_start = generated_text.find("answer:") + len("answer:")
        if answer_start < len("answer:"):  # If "answer:" not found
            answer_start = generated_text.find(query) + len(query)
            
        # Remove everything after the first question mark or period
        end_markers = ["?", "Question:", "Context:"]
        end_positions = [generated_text.find(marker, answer_start) for marker in end_markers]
        end_positions = [pos for pos in end_positions if pos != -1]
        
        if end_positions:
            trimmed_text = generated_text[answer_start:min(end_positions)].strip()
        else:
            trimmed_text = generated_text[answer_start:].strip()

        # Sentence cleaning
        sentences = [s.strip() for s in re.split(r'[.!?]+', trimmed_text) if s.strip()]
        unique_sentences = []
        for sentence in sentences:
            s_norm = sentence.lower()
            if s_norm and s_norm not in [u.lower() for u in unique_sentences]:
                unique_sentences.append(sentence)

        # Remove consecutive repeated words
        words = ' '.join(unique_sentences).split()
        deduped_words = []
        for w in words:
            if not deduped_words or w.lower() != deduped_words[-1].lower():
                deduped_words.append(w)

        cleaned_answer = ' '.join(deduped_words).strip()
        
        if not cleaned_answer:
            return "Could not generate a valid answer."
        
        # Ensure complete sentence
        if cleaned_answer and not is_complete_sentence(cleaned_answer):
            cleaned_answer = cleaned_answer[0].upper() + cleaned_answer[1:]
            if cleaned_answer[-1] not in '.!?':
                cleaned_answer += '.'

        print("Question:", query)
        print("Context:", context)
        print("Answer:", cleaned_answer)
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "An error occurred while generating the response."

# Test query
query = "What is AI?"
lightweight_rag(query)
