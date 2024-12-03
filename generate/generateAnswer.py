def run_inference(client,
                  model,
                  file_base64,
                  question,
                  file_type,
                  history,
                  retrieved_chunks,
                  max_history=4):
    """사용자 질문에 대해 이미지, pdf 등을 보고 답변"""

    # 메시지 초기화
    messages = []

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
    print("Chat History")
    print()
    for message in messages:
        print(message)
        print()
    print('-' * 20)

    # 파일 타입에 맞춰 프롬프트 생성
    if file_type == "image":
        content = f"data:image/jpeg;base64,{file_base64}"
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", 
                "text": """
                You are an AI architecture assistant. Analyze the provided AI model diagram and answer these questions:

                1. Based on the model architecture shown, what is the nature of the input data and how is it processed initially?
                2. What are the key components of the architecture?
                3. According to the image, what is the output of the model and how is it generated from the components?
                
                Please provide clear and concise answers.
                """},
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
        # prompt of logical puzzle in text
        prompt = f"""
        You are a logical puzzle expert. Solve the given puzzle step-by-step using logical reasoning. Provide a concise and clear final answer.

        **Instructions:**
        1. Break down the problem logically.
        2. Solve step-by-step to ensure clarity.
        3. Provide the final answer based on your reasoning.

        **Puzzle:** {question}
        """
        messages.append({"role": "user", "content": prompt})

    # OpenAI API 호출
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
    )

    # 결과 반환
    result = chat_completion.choices[0].message.content
    return result
