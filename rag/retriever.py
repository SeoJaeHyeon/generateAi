import spacy
from sentence_transformers import SentenceTransformer
import faiss

def chunk_text_with_overlap(text, chunk_size=200, overlap_size=50):
    """
    Splits text into chunks of a specified number of characters with overlap.
    """
    nlp_ko = spacy.load("en_core_web_md")
    doc = nlp_ko(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for token in doc:
        token_length = len(token.text_with_ws)
        # If the current chunk reaches its size limit, we add it to chunks
        if current_length + token_length > chunk_size:
            # Add the current chunk to the list
            chunks.append("".join(current_chunk).strip())
            # Keep the last `overlap_size` characters for the next chunk
            overlap_chunk = "".join(current_chunk)[-overlap_size:]
            # Reset the current chunk and current length
            current_chunk = [overlap_chunk]
            current_length = len(overlap_chunk)
        # Add the token to the current chunk
        current_chunk.append(token.text_with_ws)
        current_length += token_length

    # Add the final chunk if it exists
    if current_chunk:
        chunks.append("".join(current_chunk).strip())

    return chunks

def embedding_to_vector_store(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    # Generate embeddings for the corpus
    corpus_embeddings = model.encode(chunks)

    # Initialize a FAISS index
    dimension = corpus_embeddings.shape[1]
    print(f"Dimension of the embeddings: {dimension}")
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance metric
    index.add(corpus_embeddings)  # Add embeddings to the index
    
    return model, index

def retrieve_augmented_generation(chunks, sentence_embed_model, index, question, k=3):
    
    query_embedding = sentence_embed_model.encode([question])
    _, indices = index.search(query_embedding, k)
    
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    
    return retrieved_chunks
    