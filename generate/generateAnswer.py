def run_inference(client,
                  model,
                  file_base64,
                  question,
                  file_type,
                  history,
                  retrieved_chunks,
                  max_history = 4):
    """Run inference based on file (image/PDF) and user query."""
    
       # 서버로 넘길 메시지 구성
       
       
       
    system_prompt = '''
    You are an assistant. 
    When you have a previous conversation and an up-to-date question, it may or may not be related to each other. 
    If it is relevant, please remember and answer the previous conversation, 
    and if it is not, ignore the previous conversation and answer it.
    '''
    messages = [{"role": "system", "content": system_prompt}]
    
    
    # 이전 채팅 기록 추가
    if len(history) > max_history:
        for user_message, assistant_response in history[-max_history:]:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_response})
    else:
        for user_message, assistant_response in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_response})
    
    
    print()
    print("chat history")
    print()
    for message in messages:
        print(message)
        print()
    print('-'*20)    
    
    
    
    # 파일 콘텐츠 준비
    if file_type == "image":
        content = f"data:image/jpeg;base64,{file_base64}"
        messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", 
                     "image_url": {"url": content}},
                ],
            })
        
    elif file_type == "pdf":
        content = file_base64  # PDF 텍스트 내용
        
        messages.append({"role": "user", 
                         "content": f"{question}\n\nFile Content:\n{retrieved_chunks}"})

    elif file_type == 'text':
        messages.append({"role": "user", 
                         "content": f"{question}"})
    
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
    )
    
    result = chat_completion.choices[0].message.content 
    
    return result
