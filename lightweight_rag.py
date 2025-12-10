import numpy as np
import re
import sys
import time
import threading
import itertools
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    pipeline,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Configure model and device options
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # small instruct; swap to "microsoft/phi-2" for lighter
DEVICE_MODE = "auto"  # options: "auto", "mps", "cuda"

def get_device_map(mode: str):
    if mode == "mps":
        return {"": "mps"}   # Apple GPU
    if mode == "cuda":
        return {"": "cuda"}  # Nvidia GPU
    return "auto"            # auto-select (MPS on Apple, CUDA on Nvidia, CPU fallback)

# Initialize the sentence-transformers model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact and fast for embeddings

# Sample corpus for retrieval
corpus = [
    "Python is a popular programming language.",
    "AI stands for Artificial Intelligence.",
    "RAG models combine retrieval and generation for better context.",
    "Transformers use self-attention to capture relationships between tokens across a sequence.",
    "Vector databases store embeddings to enable fast similarity search over unstructured text.",
    "Few-shot prompting provides exemplars so a language model can mimic the pattern for new inputs.",
    "Mistral-7B-Instruct is a compact instruction-tuned model suitable for on-device inference.",
    "Apple Silicon (MPS) can accelerate PyTorch models; set device_map to mps or auto on macOS.",
    "Quantisation (e.g., 4-bit) reduces model memory footprint with minor quality trade-offs.",
    "Caching model weights locally avoids repeated downloads and speeds up subsequent runs."
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
    if not text:
        return "No valid answer generated. Run script again."

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
    
    if not truncated_text:
        return "No complete sentences found in response. Run script again."

    return truncated_text.strip()

def make_model_and_tokenizer(model_name, device_map):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 4-bit for CUDA via bitsandbytes; fp16 otherwise (e.g., MPS/auto on Mac)
    if device_map == {"": "cuda"}:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quant_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype="auto",  # fp16/bf16 on MPS/CPU where supported
        )
    return tokenizer, model

def spinner(text="Generating..."):
    stop = threading.Event()
    def run():
        for ch in itertools.cycle("|/-\\"):
            if stop.is_set():
                break
            sys.stdout.write(f"\r{text} {ch}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(text) + 2) + "\r")
    return stop, threading.Thread(target=run, daemon=True)

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.2f}s"

# Full RAG pipeline
def lightweight_rag(query):
    # Define the maximum number of tokens to generate
    max_new_tokens = 150

    # Retrieve the most relevant context
    context = retrieve(query, corpus, corpus_embeddings)

    # Initialize tokenizer and model (Apple Silicon- / CUDA-friendly)
    device_map = get_device_map(DEVICE_MODE)
    tokenizer, model = make_model_and_tokenizer(MODEL_NAME, device_map)

    # Generate a response using the chosen model
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
        torch_dtype="auto",
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    stop, t = spinner("Generating")
    t.start()
    start_time = time.time()
    try:
        # Generate response
        response = generator(
            f"Context: {context}\nQuestion: {query}\nAnswer:",
            max_new_tokens=max_new_tokens,   # cap generation
            max_length=None,                  # rely on max_new_tokens
            num_return_sequences=1,
            repetition_penalty=1.2,
            length_penalty=1.0,
        )
    finally:
        stop.set()
        t.join()
    duration = time.time() - start_time

    try:
        generated_text = response[0]['generated_text']
        answer_start = generated_text.find("Answer:") + len("Answer:")
        if answer_start < len("Answer:"):
            print("Could not generate a valid answer.")
            return
        answer = generated_text[answer_start:].strip()
        cleaned_answer = clean_output(answer)
        truncated_answer = truncate_at_sentence_end(cleaned_answer, max_new_tokens)
        print("Question:", query)
        print("Context:", context)
        print("Answer:", truncated_answer)
        print(f"Inference time: {format_duration(duration)}")
    except Exception as e:
        print(f"Error generating response: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Lightweight RAG CLI")
    parser.add_argument("-q", "--query", help="Question to ask (will query the corpus for RAG).")
    args = parser.parse_args()

    query = args.query
    if not query:
        try:
            query = input("Enter your question (will query the corpus for RAG): ").strip()
        except KeyboardInterrupt:
            print("\nAborted.")
            return
    if not query:
        print("No question provided.")
        return
    lightweight_rag(query)

# Entry point
if __name__ == "__main__":
    main()