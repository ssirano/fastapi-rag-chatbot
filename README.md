FastAPI RAG 문서 챗봇
RAG(Retrieval-Augmented Generation)를 활용한, 문서 기반 질의응답 챗봇입니다. 문서를 업로드하면 해당 내용을 분석하여 사용자 질문에 답변합니다.
소개
이 프로젝트는 다양한 문서(PDF, TXT, DOCX)를 업로드하고 내용에 대해 질문할 수 있는 챗봇 시스템입니다. 문서의 내용을 벡터화하여 저장하고, 질문과 관련된 정보를 검색한 후 생성형 AI를 통해 답변을 제공합니다.
주요 기능:

다양한 포맷의 문서 처리(PDF, TXT, DOCX)
문서 내용 청크 분할 및 벡터화
의미 기반 검색
자연스러운 답변 생성

기술 스택

백엔드: FastAPI
임베딩 모델: SentenceTransformer
언어 모델: Google Gemma (최신 모델 사용)
벡터 검색: 코사인 유사도 기반 검색
프론트엔드: HTML, CSS, JavaScript

환경 설정
요구사항

Python 3.8 이상
최소 8GB RAM 권장 (임베딩 모델 로드를 위해)

사용 방법

웹 인터페이스에 접속합니다.
LLM 옵션을 선택합니다:

HuggingFace API (무료)
HuggingFace API (API 키 사용)
로컬 폴백 알고리즘


분석할 문서를 업로드합니다.
문서에 관한 질문을 입력합니다.
답변을 확인합니다.

트러블슈팅
1. HuggingFace API 오류
문제: 모델 API가 "Service Unavailable(503)" 또는 기타 오류를 반환합니다.
해결방법:

시스템이 자동으로 로컬 폴백 알고리즘으로 전환됩니다.
HuggingFace API에 요청이 너무 많이 가지 않도록 지연 시간을 조정해 볼 수 있습니다.
API 키를 사용하면 더 안정적인 서비스를 받을 수 있습니다.

2. 메모리 부족 오류
문제: 대용량 문서 처리 시 "MemoryError" 발생합니다.
해결방법:

app/rag_chatbot.py 파일에서 청크 크기(chunk_size) 값을 줄입니다.
배치 크기(batch_size)를 조정합니다.
더 작은 임베딩 모델을 사용하도록 설정합니다.

3. 한국어 문서 처리 문제
문제: 한국어가 포함된 문서의 처리 품질이 영어보다 낮을 수 있습니다.
해결방법:

app/rag_chatbot.py에서 다국어 모델로 변경

향후 계획
현재 이 프로젝트는 Google Gemma 모델을 사용하고 있으며, 아직 완벽하게 최적화되지는 않았습니다. 시간이 될 때마다 다음과 같은 개선 사항을 추가할 예정입니다:

벡터 검색 알고리즘 최적화
한국어 처리 성능 개선
대화 기록 유지 기능
