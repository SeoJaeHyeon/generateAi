import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')            # sentenceEmbedding model 선언

def chunk_text_with_overlap(text, chunk_size=800, overlap_size=200):
    """
    PDF 전체 text를 chunk 단위로 분할하는데, 같은 문장이 잘릴 수도 있으므로, 중첩하여 분할
    """
    nlp_ko = spacy.load("en_core_web_md")
    doc = nlp_ko(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for token in doc:
        token_length = len(token.text_with_ws)
        if current_length + token_length > chunk_size:
            chunks.append("".join(current_chunk).strip())                       # chunk_size 만큼 추가

            overlap_chunk = "".join(current_chunk)[-overlap_size:]              # 중첩할 토큰 지정

            current_chunk = [overlap_chunk]                                     # 다음 청크 준비
            current_length = len(overlap_chunk)

        current_chunk.append(token.text_with_ws)
        current_length += token_length

   
    if current_chunk:
        chunks.append("".join(current_chunk).strip())                           # 마지막 남은 청크들도 포함

    return chunks

def batch_encode(chunks, model, batch_size=16):
    ''' 청크를 임베딩할 때 Out of Memory 발생 방지를 위해 batch 별로 문장 임베딩 '''                                 
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings.append(model.encode(batch))
    return np.vstack(embeddings)


def embedding_to_vector_store(chunks):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    global model
    corpus_embeddings = batch_encode(chunks=chunks,                             # 문장 임베딩 수행
                                     model=model)


    dimension = corpus_embeddings.shape[1]
    print(f"Dimension of the embeddings: {dimension}")
    index = faiss.IndexFlatL2(dimension)                                        # L2 distance metric로 index 초기화
    index.add(corpus_embeddings)                                                # 임베딩을 faiss vectorstore에 저장
    
    return model, index

def retrieve_augmented_generation(chunks, sentence_embed_model, index, question, k=3):
    
    query_embedding = sentence_embed_model.encode([question])                   # 질문에 대한 임베딩 수행
    _, indices = index.search(query_embedding, k)                               # 질문과 유사한 청크들 k개 가져옴
    
    retrieved_chunks = [chunks[idx] for idx in indices[0]]                      # 원래 청크들에서 index를 활용해 가져옴
    
    return retrieved_chunks
    