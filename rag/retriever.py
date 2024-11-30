import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch 

# NLTK의 토크나이저를 사용
nltk.download('punkt_tab')

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')            # sentenceEmbedding model 선언
reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
reranker_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')


def chunk_text_with_overlap(text, summary, chunk_size=500, overlap_size=200):
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
        chunks.append("".join(current_chunk).strip())

    final_chunk = []
    for idx, chunk in enumerate(chunks):
        # 요약과 함께 컨텍스트 청크 생성
        context_chunk = f"""
        <summary>
        {summary}
        </summary>
        <chunk>
        {chunk}
        </chunk>
        """
        final_chunk.append(context_chunk)
    return final_chunk

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
    
# 텍스트를 토큰화하는 함수
def tokenize(text):
    return word_tokenize(text.lower())  # 소문자로 변환하여 토큰화

# 텍스트를 청크로 나누는 함수
def chunk_text_with_overlap_nltk(text, summary, chunk_size=500, overlap_size=200):
    context_chunk = f"""
    <summary>
    {summary}
    </summary>
    <chunk>
    """
    tokens = tokenize(text)  # 먼저 토큰화
    context_tokens = tokenize(context_chunk)

    chunks = []
    current_chunk = []
    
    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= chunk_size:
            chunk_with_context = context_tokens + current_chunk
            chunks.append(chunk_with_context)
            current_chunk = current_chunk[-overlap_size:]  # 중첩된 부분만 남김

    if current_chunk:  # 남아있는 청크도 추가
        chunk_with_context = context_tokens + current_chunk
        chunks.append(chunk_with_context)
    
    return chunks

# BM25 인덱스를 빌드하는 함수
def build_bm25_index(chunks):
    tokenized_chunks = [tokenize(" ".join(chunk)) for chunk in chunks]  # 청크를 먼저 토큰화
    bm25 = BM25Okapi(tokenized_chunks)  # BM25 모델을 생성
    return bm25

# 쿼리와 유사한 청크를 찾는 함수
def retrieve_relevant_chunks(chunks, bm25, question, top_k=3):
    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)  # 쿼리와 각 청크의 유사도 점수 계산
    ranked_chunks = np.argsort(scores)[::-1]  # 높은 점수 순으로 정렬
    top_chunks = [chunks[i] for i in ranked_chunks[:top_k]]  # 상위 K개 청크 반환
    for idx, chunk in enumerate(top_chunks):
        top_chunks[idx] = ' '.join(chunk)

    return top_chunks

def create_pairs(question, embed_retrieved_chunks, bm25_retrieved_chunks):
    pairs = []
    for i in range(3):
        pairs.append([question, embed_retrieved_chunks[i]])
        pairs.append([question, bm25_retrieved_chunks[i]])
    return pairs 

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def reranker(question, embed_retrieved_chunks, bm25_retrieved_chunks):
    global reranker_tokenizer, reranker_model

    pairs = create_pairs(question=question,
                         embed_retrieved_chunks=embed_retrieved_chunks,
                         bm25_retrieved_chunks=bm25_retrieved_chunks)

    reranker_model.eval()
    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = exp_normalize(scores.numpy())
    
    top_indices = np.argsort(scores)[-3:]
    final_context = []
    for idx in top_indices:
        final_context.append(pairs[idx][1]) 
    return final_context
