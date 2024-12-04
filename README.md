# generateAi
**생성형 AI 응용 프로젝트 팀 D 서재현, 전지훈, 서재연**

# 1. vllm 실행 방법
```
serve llava-hf/llava-1.5-7b-hf --chat-template /vllm/template_llava.jinja --gpu-memory-utilization 0.6 --api-key token-abc123 --port 8000
```

# 2. 프로젝트 주제
- Text, image, document를 주고, 그에 대해 QA 할 수 있는 시스템

# 3. 프로젝트 개요
- vLLM을 이용해 Multimodal Large Language Model 모델 serving
- 채팅이 가능하도록 이전 대화의 history를 모델에 반영
- text에 대해 chain of thought 등과 같은 prompt engineering을 통해 답변 성능 향상
- image로부터 얻고 싶은 정보에 대하여 prompt engineering을 통해 답변 성능 향상
- user의 query에 대하여 document로부터 contextual chunking, embedding vector store, TF-IDF index, reranker를 이용한 retriever

# 4. text prompt
```
"You are a logical puzzle expert. Solve the given puzzle step-by-step using logical reasoning. Provide a concise and clear final answer.

Instructions:

1. Break down the problem logically.
 ⇒ 문제를 논리적으로 분해하세요.
2. Solve step-by-step to ensure clarity.  
 ⇒ 명확성을 위해 단계별로 해결하세요.
3. Provide the final answer based on your reasoning.
 ⇒ 당신의 추론에 기반한 최종 답변을 제공하세요.

Puzzle: {question}"
```

# 5. image prompt
```
'''
You are an AI architecture assistant. (AI Architecture Assistant 역할 부여  -> 기술적이고 전문적인 답변을 기대)
Analyze the provided AI model diagram and provide a comprehensive explanation 
that includes the nature of the input data based on the model architecture shown 
and how it is processed initially, the key components of the architecture, 
and according to the image, what the output of the model is and how it is generated from the components.

Please present your answer as a single, cohesive paragraph without numbering or separating the points.
{question}
'''
 '''
 - 구체적인 분석 요청
 - 프롬프트는 다음의 구체적인 분석 지시사항을 포함
 - 입력 데이터의 특성과 모델 아키텍처의 초기 처리 방식 설명
 - 모델 아키텍처의 주요 구성 요소 설명
 - 이미지에 나타난 내용을 기반으로 모델 출력의 형태와 생성 과정을 설명
 - CoT(Chain-of-Thought) 적용
- 프롬프트를 통해 체계적으로 문제를 분석하고 답변을 구성하도록 유도
- 입력 데이터의 특성과 초기 처리 방식 분석
- 아키텍처 구성 요소의 역할과 관계 설명
- 출력이 생성되는 과정 설명
-이를 통해 모델이 한 번에 답을 생성하기보다는, 각 단계를 논리적으로 분석하고 연결 지어 답을 생성하도록 설계
'''
```

# 6. RAG Pipeline
![rag pipeline](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8f82c6175a64442ceff4334b54fac2ab3436a1d1-3840x2160.png&w=3840&q=75)

- Spacy를 이용해 토큰 단위로 overlap 하여 chunking
- 각 chunk에 대하여 문서의 context를 삽입하여 contextual chunking
- sentence transformer를 이용해 embedding 후 faiss vector store에 indexing
- TF-IDF 기반의 BM25를 이용해 indexing
- user의 query에 기반하여 embedding과 TF-IDF 각각에 대해 유사도 기준 top-3 추출
- 총 6개의 chunk에 대해 reranker를 이용해 질문과 가장 유의미한 최종 chunk top-3 선정
- 해당 chunk들을 model의 context에 넘겨주고, user query에 response