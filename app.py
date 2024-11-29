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

# OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"  # Example API base

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id
print(f"Using model: {model}")

chunks, sentence_embed_model, index = None, None, None

def generate_response(file, question, history):
    """Generate a response based on the uploaded file and question."""
    
    global chunks, sentence_embed_model, index
    
    if file is None:
        chunks, sentence_embed_model, index = None, None, None
        
        file_type = "text"
        file_base64 = None  # 텍스트만 있을 경우, 파일 내용은 없음
        response = run_inference(client,
                                 model, 
                                 file_base64,
                                 question,
                                 file_type,
                                 history,
                                 None,
                                 4)
    else:   
        file_type = "image" if file.name.endswith((".jpg", ".jpeg", ".png")) else "pdf"
        if file_type == "image":
            file_base64 = encode_base64_content_from_file(file.name)
            response = run_inference(client, 
                                     model,
                                     file_base64, 
                                     question, 
                                     file_type,
                                     history, 
                                     None,
                                     4)
        elif file_type == "pdf":
            file_base64 = extract_text_from_pdf(file.name)
            if index is None:
                print("Pdf Text 추출 완료")
                chunks = chunk_text_with_overlap(text=file_base64)
                print("Pdf text chunking 완료")
                sentence_embed_model, index = embedding_to_vector_store(chunks=chunks)
                
            retrieved_chunks = retrieve_augmented_generation(chunks=chunks,
                                                             sentence_embed_model=sentence_embed_model,
                                                             index=index,
                                                             question=question,
                                                             k=3)
            print("문서에서 질문과 연관된 Top K context 추출 완료")
            print("context :")
            print(retrieved_chunks)
            print()
            response = run_inference(client, 
                                     model,
                                     file_base64, 
                                     question, 
                                     file_type, 
                                     history,
                                     retrieved_chunks,
                                     4)

    history.append((question, ""))
    words = response.split()  # 응답을 단어별로 나누기
    for word in words:
        # 현재 단어를 히스토리에 추가하며 업데이트
        history[-1] = (question, history[-1][1] + " " + word)
        yield history
        time.sleep(0.1)  # 단어 출력 간 지연 시간 설정

    
# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    gr.Markdown("# File Analysis Chatbot (PDF & Image Support)")

    with gr.Row():
        # 이미지와 PDF를 좌우로 배치
        with gr.Column():
            image_preview = gr.Image(label="Image Preview")
        with gr.Column():
            pdf_preview = gr.HTML(label="PDF Preview")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload a File (PDF/Image)")
            text_input = gr.Textbox(label="Ask a Question", placeholder="Enter your question")
            submit_button = gr.Button("Submit")

    chat_history = gr.Chatbot(label="Chat History")

    file_input.change(preview_file, inputs=[file_input], outputs=[image_preview, pdf_preview])
    submit_button.click(
        generate_response,  # 제너레이터 함수 사용
        inputs=[file_input, text_input, chat_history],
        outputs=[chat_history],  # 출력은 한 개의 히스토리로
    )

demo.queue().launch(share=False)



