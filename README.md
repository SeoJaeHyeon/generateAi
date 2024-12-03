# generateAi
**생성형 AI 응용 프로젝트 팀 D 서재현, 전지훈, 서재연**

# vllm 실행 방법
```
serve llava-hf/llava-1.5-7b-hf --chat-template $TEMPLATE_PATH --gpu-memory-utilization 0.6 --api-key token-abc123 --port 8000
```

# 프로젝트 주제
- Text, image, document를 주고, 그에 대해 QA 할 수 있는 시스템

# 프로젝트 개요
- vLLM을 이용해 Multimodal Large Language Model 모델 serving
- 채팅이 가능하도록 이전 대화의 history를 모델에 반영
- text에 대해 chain of thought 등과 같은 prompt engineering을 통해 답변 성능 향상
- image로부터 얻고 싶은 정보에 대하여 prompt engineering을 통해 답변 성능 향상
- user의 query에 대하여 document로부터 contextual chunking, embedding vector store, TF-IDF index, reranker를 이용한 retriever

# text prompt
```
"You are a logical puzzle expert. Solve the given puzzle step-by-step using logical reasoning. Provide a concise and clear final answer.

**Instructions:**

1. Break down the problem logically.
 ⇒ 문제를 논리적으로 분해하세요.
2. Solve step-by-step to ensure clarity.  
 ⇒ 명확성을 위해 단계별로 해결하세요.
3. Provide the final answer based on your reasoning.
 ⇒ 당신의 추론에 기반한 최종 답변을 제공하세요.

**Puzzle:** {question}"
```

# image prompt
```
"You are an AI architecture assistant. Analyze the provided AI model diagram and answer these questions:

1. Based on the model architecture shown, what is the nature of the input data and how is it processed initially?
 ⇒ 표시된 모델 아키텍처를 기반으로, 입력 데이터의 성질은 무엇이며 초기에는 어떻게 처리되는가?
2. What are the key components of the architecture?
 ⇒ 아키텍처의 주요 구성 요소가 무엇인가?
3. According to the diagram, what is the output of the model and how is it generated from the components?
 ⇒ 이미지에 따르면, 모델의 출력은 무엇이며 그 구성 요소로부터 어떻게 생성되는가?

Please provide clear and concise answers."
```

# RAG Pipeline
![rag pipeline](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8f82c6175a64442ceff4334b54fac2ab3436a1d1-3840x2160.png&w=3840&q=75)

