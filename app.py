import gradio as gr
from openai import OpenAI

from inputFile.inputFile import encode_base64_content_from_file 
from inputFile.inputFile import extract_text_from_pdf
from inputFile.inputFile import preview_file
from generate.generateAnswer import run_inference

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



def generate_response(file, question, history):
    """Generate a response based on the uploaded file and question."""
    if file is None:
        file_type = "text"
        file_base64 = None  # 텍스트만 있을 경우, 파일 내용은 없음
        response = run_inference(client, model, file_base64, question, file_type, history)
        history.append((question, response))
        return history, history

    else:   
        file_type = "image" if file.name.endswith((".jpg", ".jpeg", ".png")) else "pdf"
        if file_type == "image":
            file_base64 = encode_base64_content_from_file(file.name)
            response = run_inference(client, model, file_base64, question, file_type, history)
            history.append((question, response))
            return history, history

        elif file_type == "pdf":
            file_base64 = extract_text_from_pdf(file.name)
            response = run_inference(client, model, file_base64, question, file_type, history)
            history.append((question, response))
            return history, history

    

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
        generate_response,
        inputs=[file_input, text_input, chat_history],
        outputs=[chat_history, chat_history],
    )

demo.queue().launch(share=False)
