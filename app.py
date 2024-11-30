import gradio as gr
from openai import OpenAI
import time
from inputFile.inputFile import encode_base64_content_from_file 
from inputFile.inputFile import extract_text_from_pdf
from inputFile.inputFile import preview_file
from generate.generateAnswer import run_inference
from rag.retriever import chunk_text_with_overlap
from rag.retriever import embedding_to_vector_store
from rag.retriever import retrieve_augmented_generation
from rag.retriever import chunk_text_with_overlap_nltk
from rag.retriever import build_bm25_index
from rag.retriever import retrieve_relevant_chunks
from rag.retriever import reranker

# OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"  # Example API base

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id                                                       # llava-hf/llava-1.5-7b-hf
print(f"Using model: {model}")

chunks, sentence_embed_model, index = None, None, None                          # pdf 내용을 저장할 vectorstore 
chunks_nltk, bm25, = None, None 

def generate_response(file, summary ,question, history):
    ''' 질문에 대한 응답을 내는 함수 '''
    
    global chunks, sentence_embed_model, index
    global chunks_nltk, bm25
    
    if file is None:                                                            # 이미지나 pdf 입력이 안된 경우
        file_type = "text"
        response = run_inference(client,
                                 model, 
                                 None,
                                 question,
                                 file_type,
                                 history,
                                 None,
                                 4)
    else:   
        file_type = "image" if file.name.endswith((".jpg", ".jpeg", ".png")) else "pdf"
        if file_type == "image":                                                # 이미지가 입력된 경우
            file_base64 = encode_base64_content_from_file(file.name)
            response = run_inference(client, 
                                     model,
                                     file_base64, 
                                     question, 
                                     file_type,
                                     history, 
                                     None,
                                     4)
        elif file_type == "pdf":                                                # pdf가 입력된 경우
            file_base64 = extract_text_from_pdf(file.name)                      # pdf로 부터 텍스트 추출
            if index is None:
                chunks = chunk_text_with_overlap(text=file_base64,
                                                 summary=summary)                   # 추출된 텍스트를 chunk로 분할 
                      
                sentence_embed_model, index = embedding_to_vector_store(chunks=chunks)  # chunk 들을 faiss에 인덱싱
                
            retrieved_chunks = retrieve_augmented_generation(chunks=chunks,     
                                                             sentence_embed_model=sentence_embed_model,
                                                             index=index,
                                                             question=question,
                                                             k=3)                        # faiss로 부터 질문과 유사한 pdf chunks k개 반환


            if bm25 is None:
                chunks_nltk = chunk_text_with_overlap_nltk(text=file_base64,
                                                           summary=summary)
                bm25 = build_bm25_index(chunks=chunks_nltk)
            
            bm25_retrieved_chunks = retrieve_relevant_chunks(chunks=chunks_nltk,
                                                             bm25=bm25,
                                                             question=question,
                                                             top_k=3)

            final_context = reranker(question=question,
                                     embed_retrieved_chunks=retrieved_chunks,
                                     bm25_retrieved_chunks=bm25_retrieved_chunks)

            response = run_inference(client, 
                                     model,
                                     file_base64, 
                                     question, 
                                     file_type, 
                                     history,
                                     final_context,
                                     4)                                                 

    history.append((question, ""))

    words = response.split()                                                    # 응답을 단어별로 나누기(GPT 처럼)
    for word in words:                                                           
        history[-1] = (question, history[-1][1] + " " + word)                   # 현재 단어를 히스토리에 추가하며 업데이트
        yield history
        time.sleep(0.1)                                                         # 단어 출력 간 지연 시간 설정


def clear_file():
    """pdf를 삭제하는 경우 vectorstore 초기화."""     
    global chunks, sentence_embed_model, index
    global chunks_nltk, bm25
    chunks, sentence_embed_model, index = None, None, None
    chunks_nltk, bm25 = None
    return None, '<div style="border:1px solid #e0e0e0; padding:10px; height:235px;">PDF Preview</div>' 
    
# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    gr.Markdown("# 생성형 AI 응용 ( RAG System with Open Source - Team D 서재현, 전지훈, 서재연 )")

    with gr.Row():
        # 이미지와 PDF를 좌우로 배치
        with gr.Column():
            image_preview = gr.Image(label="Image Preview")
        with gr.Column():
            pdf_preview = gr.HTML(label="PDF Preview",value='<div style="border:1px solid #e0e0e0; padding:10px; height:235px;">PDF Preview</div>')
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload a File (PDF/Image)")
            summary_input = gr.Textbox(label="Summary of the file")
            text_input = gr.Textbox(label="Ask a Question", placeholder="Enter your question")
            submit_button = gr.Button("start")

    chat_history = gr.Chatbot(label="Chat History")

    file_input.change(preview_file, inputs=[file_input], outputs=[image_preview, pdf_preview])
    file_input.clear(clear_file, outputs=[image_preview, pdf_preview])  # X 버튼 동작

    
    submit_button.click(
        generate_response,  # 제너레이터 함수 사용
        inputs=[file_input, summary_input, text_input, chat_history],
        outputs=[chat_history],  # 출력은 한 개의 히스토리로
    )

demo.queue().launch(share=True)