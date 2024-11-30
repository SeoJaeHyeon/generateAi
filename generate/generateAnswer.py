

def run_inference(client,
                  model,
                  file_base64,
                  question,
                  file_type,
                  history,
                  retrieved_chunks,
                  max_history = 4):
    
    """사용자 질문에 대해 이미지, pdf 등을 보고 답변"""
    
    # 모델에 역할 부여
    system_prompt = '''
    You are an assistant. 
    When you have a previous conversation and an up-to-date question, it may or may not be related to each other.
    If it is relevant, please refer to the previous conversation and reply,
    If not, ignore the previous conversation and answer.
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
    
    
    
    # 파일 타입에 맞춰 프롬프트 생성
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
        prompt = f"""
        You should answer the given papers and questions accurately and in detail.
        If user ask about the algorithm of the paper,
        please answer by thinking of the operation process of the algorithm as step by step based on the given context.
        <question>
        {question}
        </question>
        <context>
        {retrieved_chunks}
        </context>
        """        
        messages.append({"role": "user", 
                         "content": prompt})

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
