import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
import asyncio
import torch
import time
from pathlib import Path
import tempfile

# 파일 처리 라이브러리
import PyPDF2
import docx
import fitz  # PyMuPDF

# 텍스트 전처리 및 임베딩
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# 벡터 DB 역할
from sentence_transformers import SentenceTransformer

# API 요청
import requests

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NLTK 리소스 다운로드 (첫 실행 시 필요)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class DocumentProcessor:
    """문서 처리 클래스: 다양한 형식의 문서를 읽고 텍스트로 변환"""
    
    def __init__(self):
        logger.info("DocumentProcessor 초기화")
        
    async def read_document_async(self, file_path: str) -> str:
        """비동기: 파일 확장자에 따라 적절한 방법으로 문서 읽기"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            return await self._read_pdf_async(file_path)
        elif ext == '.txt':
            return await self._read_txt_async(file_path)
        elif ext in ['.docx', '.doc']:
            return await self._read_docx_async(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")
    
    def read_document(self, file_path: str) -> str:
        """동기: 파일 확장자에 따라 적절한 방법으로 문서 읽기"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            return self._read_pdf(file_path)
        elif ext == '.txt':
            return self._read_txt(file_path)
        elif ext in ['.docx', '.doc']:
            return self._read_docx(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")
    
    async def _read_pdf_async(self, file_path: Path) -> str:
        """비동기: PDF 파일 읽기"""
        # 비동기 IO 작업을 위해 실행 루프에서 동기 함수 실행
        return await asyncio.to_thread(self._read_pdf, file_path)
    
    def _read_pdf(self, file_path: Path) -> str:
        """PDF 파일 읽기"""
        try:
            text = ""
            # PyMuPDF 사용 (더 나은 성능)
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            logger.info(f"PDF 파일 성공적으로 읽음: {file_path.name}, 크기: {len(text)} 문자")
            return text
        except Exception as e:
            logger.error(f"PDF 파일 읽기 오류: {str(e)}")
            # 폴백: PyPDF2로 시도
            try:
                text = ""
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                logger.info(f"PyPDF2로 PDF 파일 읽음: {file_path.name}")
                return text
            except Exception as e2:
                logger.error(f"PyPDF2로도 PDF 읽기 실패: {str(e2)}")
                raise
    
    async def _read_txt_async(self, file_path: Path) -> str:
        """비동기: 텍스트 파일 읽기"""
        try:
            async with open(file_path, 'r', encoding='utf-8') as file:
                text = await file.read()
            logger.info(f"텍스트 파일 성공적으로 읽음: {file_path.name}")
            return text
        except UnicodeDecodeError:
            # UTF-8로 읽기 실패시 다른 인코딩 시도
            encodings = ['cp949', 'euc-kr', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    logger.info(f"텍스트 파일을 {encoding} 인코딩으로 읽음: {file_path.name}")
                    return text
                except UnicodeDecodeError:
                    continue
            logger.error(f"텍스트 파일 인코딩 감지 실패: {file_path.name}")
            raise
    
    def _read_txt(self, file_path: Path) -> str:
        """텍스트 파일 읽기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"텍스트 파일 성공적으로 읽음: {file_path.name}")
            return text
        except UnicodeDecodeError:
            # UTF-8로 읽기 실패시 다른 인코딩 시도
            encodings = ['cp949', 'euc-kr', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    logger.info(f"텍스트 파일을 {encoding} 인코딩으로 읽음: {file_path.name}")
                    return text
                except UnicodeDecodeError:
                    continue
            logger.error(f"텍스트 파일 인코딩 감지 실패: {file_path.name}")
            raise
    
    async def _read_docx_async(self, file_path: Path) -> str:
        """비동기: Word 문서 읽기"""
        return await asyncio.to_thread(self._read_docx, file_path)
    
    def _read_docx(self, file_path: Path) -> str:
        """Word 문서 읽기"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
            logger.info(f"DOCX 파일 성공적으로 읽음: {file_path.name}")
            return text
        except Exception as e:
            logger.error(f"DOCX 파일 읽기 오류: {str(e)}")
            raise


class TextSplitter:
    """텍스트 분할 클래스: 의미 단위로 더 효과적으로 분할"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"TextSplitter 초기화: 청크 크기={chunk_size}, 오버랩={chunk_overlap}")
    
    def split_text(self, text: str) -> List[str]:
        """텍스트를 의미 있는 청크로 분할 - 개선된 알고리즘"""
        if not text:
            logger.warning("분할할 텍스트가 비어 있습니다.")
            return []
        
        # 단락 단위 먼저 분할 (더 의미 있는 단위)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        logger.info(f"텍스트를 {len(paragraphs)}개 단락으로 분할")
        
        # 단락이 너무 길면 문장 단위로 추가 분할
        all_sentences = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                all_sentences.append(para)
            else:
                try:
                    sentences = sent_tokenize(para)
                    all_sentences.extend(sentences)
                except Exception as e:
                    logger.error(f"문장 분할 오류: {str(e)}")
                    # 길이 기반 임시 분할
                    sub_parts = []
                    for i in range(0, len(para), self.chunk_size // 2):
                        sub_parts.append(para[i:i + self.chunk_size // 2])
                    all_sentences.extend(sub_parts)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in all_sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # 현재 청크 저장
                chunks.append(" ".join(current_chunk))
                
                # 오버랩을 위해 일부 문장 유지 (마지막 2개 문장 또는 오버랩 크기 이내)
                overlap_sentences = []
                overlap_size = 0
                
                for s in reversed(current_chunk[-2:]):  # 최대 2개 문장만 유지
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # 마지막 청크 저장
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # 너무 작은 청크 병합 (최소 100자 이상)
        merged_chunks = []
        current_merged = ""
        
        for chunk in chunks:
            if len(current_merged) + len(chunk) <= self.chunk_size:
                if current_merged:
                    current_merged += " " + chunk
                else:
                    current_merged = chunk
            else:
                if current_merged:
                    merged_chunks.append(current_merged)
                current_merged = chunk
        
        if current_merged:
            merged_chunks.append(current_merged)
        
        logger.info(f"최종적으로 {len(merged_chunks)}개 청크로 분할됨")  
        
        # 각 청크의 크기 로깅 (디버깅용)
        chunk_sizes = [len(c) for c in merged_chunks]
        logger.info(f"청크 크기 통계: 최소={min(chunk_sizes)}, 최대={max(chunk_sizes)}, 평균={sum(chunk_sizes)/len(chunk_sizes):.1f}")
        
        return merged_chunks


class VectorStore:
    """벡터 저장소 클래스: 문서 청크를 벡터화하고 저장"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.device = self._get_optimal_device()
        logger.info(f"VectorStore 초기화 - 모델: {model_name}, 장치: {self.device}")
        self.model = self._load_model(model_name)
        self.vectors = []
        self.documents = []
    
    def _get_optimal_device(self) -> torch.device:
        """최적의 연산 장치 선택 (MPS, CUDA, CPU)"""
        if torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS 가속 활성화")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logger.info("NVIDIA CUDA 가속 활성화")
            return torch.device("cuda")
        else:
            logger.info("CPU 모드로 동작")
            return torch.device("cpu")
    
    def _load_model(self, model_name: str) -> SentenceTransformer:
        """SentenceTransformer 모델 로드"""
        try:
            model = SentenceTransformer(model_name)
            
            # 장치 최적화
            if self.device.type != "cpu":
                try:
                    model.to(self.device)
                    logger.info(f"모델이 {self.device} 장치로 이동됨")
                except Exception as e:
                    logger.error(f"모델을 {self.device}로 이동 중 오류: {str(e)}")
                    logger.info("CPU로 폴백")
                    self.device = torch.device("cpu")
            
            return model
        except Exception as e:
            logger.error(f"모델 로딩 오류: {str(e)}")
            # 폴백: 더 작은 모델 시도
            fallback_model = "distiluse-base-multilingual-cased-v1"
            logger.info(f"폴백 모델 로드 시도: {fallback_model}")
            return SentenceTransformer(fallback_model)
    
    def add_documents(self, documents: List[str], batch_size: int = 16) -> int:
        """문서를 벡터화하여 저장 (배치 처리)"""
        if not documents:
            logger.warning("추가할 문서가 없습니다.")
            return 0
            
        self.documents.extend(documents)
        
        # 배치 처리로 메모리 효율성 개선
        vectors = []
        total_processed = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            logger.info(f"청크 배치 처리 중: {i+1}-{min(i+batch_size, len(documents))}/{len(documents)}")
            
            try:
                batch_vectors = self.model.encode(
                    batch, 
                    convert_to_tensor=True,
                    device=self.device
                )
                
                # 장치 간 이동 처리
                if self.device.type != "cpu":
                    batch_vectors = batch_vectors.cpu()
                
                # NumPy 변환 및 저장
                batch_vectors_np = batch_vectors.numpy()
                vectors.extend(batch_vectors_np)
                total_processed += len(batch)
                
            except Exception as e:
                logger.error(f"배치 벡터화 오류: {str(e)}")
                # 배치 크기를 줄여서 재시도
                if batch_size > 1:
                    smaller_batch_size = max(1, batch_size // 2)
                    logger.info(f"배치 크기 감소하여 재시도: {batch_size} -> {smaller_batch_size}")
                    # 현재 배치만 작은 크기로 처리
                    for j in range(i, min(i+batch_size, len(documents)), smaller_batch_size):
                        small_batch = documents[j:j+smaller_batch_size]
                        try:
                            small_batch_vectors = self.model.encode(
                                small_batch,
                                convert_to_tensor=True,
                                device=self.device
                            )
                            if self.device.type != "cpu":
                                small_batch_vectors = small_batch_vectors.cpu()
                            vectors.extend(small_batch_vectors.numpy())
                            total_processed += len(small_batch)
                        except Exception as e2:
                            logger.error(f"작은 배치 처리 중 오류: {str(e2)}")
                            # 개별 문서 처리 시도
                            for doc in small_batch:
                                try:
                                    single_vector = self.model.encode(
                                        [doc],
                                        convert_to_tensor=True,
                                        device=self.device
                                    )
                                    if self.device.type != "cpu":
                                        single_vector = single_vector.cpu()
                                    vectors.append(single_vector.numpy()[0])
                                    total_processed += 1
                                except:
                                    logger.error(f"문서 건너뜀: {doc[:50]}...")
        
        self.vectors.extend(vectors)
        logger.info(f"벡터 저장소에 {total_processed}개 문서 추가됨")
        
        return total_processed
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.6) -> List[Tuple[int, float, str]]:
        """쿼리와 유사한 문서 검색 - 임계값 필터링 추가"""
        if not self.vectors or not self.documents:
            logger.warning("벡터 저장소가 비어 있어 검색할 수 없습니다.")
            return []
            
        try:
            # 쿼리 벡터화
            query_vector = self.model.encode([query], convert_to_tensor=True, device=self.device)
            
            # 장치 간 이동
            if self.device.type != "cpu":
                query_vector = query_vector.cpu()
                
            query_vector = query_vector.numpy()[0]
            
            # 벡터 검색 - 배치 처리
            doc_vectors = np.array(self.vectors)
            
            # 정규화 및 코사인 유사도 계산
            query_norm = np.linalg.norm(query_vector)
            doc_norms = np.linalg.norm(doc_vectors, axis=1)
            
            # 분모가 0이 되는 것 방지
            safe_divisor = np.maximum(doc_norms * query_norm, 1e-10)
            similarities = np.dot(doc_vectors, query_vector) / safe_divisor
            
            # 임계값 이상인 결과만 선택
            qualified_indices = np.where(similarities >= threshold)[0]
            
            if len(qualified_indices) == 0:
                logger.warning(f"임계값({threshold}) 이상의 유사한 문서가 없습니다. 임계값 없이 재시도합니다.")
                # 임계값 없이 다시 시도
                top_indices = np.argsort(similarities)[::-1][:top_k]
            else:
                # 임계값을 통과한 항목 중 상위 k개 선택
                sorted_qualified = qualified_indices[np.argsort(similarities[qualified_indices])[::-1]]
                top_indices = sorted_qualified[:min(len(sorted_qualified), top_k)]
            
            results = []
            for idx in top_indices:
                results.append((int(idx), float(similarities[idx]), self.documents[idx]))
                
            logger.info(f"검색 성공: '{query[:30]}...'에 대해 {len(results)}개 결과 반환, 유사도: {[round(r[1], 3) for r in results]}")
            return results
            
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            # 간단한 폴백 메커니즘: TF-IDF 기반 유사도 검색
            try:
                logger.info("TF-IDF 폴백 검색 시도")
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(self.documents)
                query_vec = vectorizer.transform([query])
                
                similarities = (tfidf_matrix @ query_vec.T).toarray().flatten()
                top_indices = similarities.argsort()[::-1][:top_k]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0:
                        results.append((int(idx), float(similarities[idx]), self.documents[idx]))
                
                logger.info(f"TF-IDF 폴백 검색 성공: {len(results)}개 결과")
                return results
                
            except Exception as e2:
                logger.error(f"폴백 검색도 실패: {str(e2)}")
                return []


class LLMService:
    """LLM 서비스 클래스: 프롬프트 작성 및 LLM API 호출"""
   
    def __init__(self, api_key: Optional[str] = None, use_huggingface: bool = True):
        self.use_huggingface = use_huggingface
        self.api_key = api_key
        self.hf_api_url = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
        logger.info(f"LLM 서비스 초기화: 타입={'HuggingFace' if use_huggingface else '로컬 알고리즘'}")
    
    async def generate_answer_async(self, query: str, context: List[str]) -> str:
        """비동기: LLM을 이용해 질문에 대한 답변 생성"""
        return await asyncio.to_thread(self.generate_answer, query, context)
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """LLM을 이용해 질문에 대한 답변 생성 - 개선된 프롬프트"""
        
        # 컨텍스트 품질 정렬 및 필터링
        scored_contexts = []
        for i, ctx in enumerate(context):
            # 간단한 관련성 점수 계산 (실제로는 더 정교한 방법 사용 가능)
            words = set(re.findall(r'\w+', query.lower()))
            ctx_words = set(re.findall(r'\w+', ctx.lower()))
            overlap = len(words.intersection(ctx_words))
            scored_contexts.append((ctx, overlap / max(1, len(words)), i))
        
        # 관련성 점수로 정렬
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        filtered_contexts = [c[0] for c in scored_contexts if c[1] > 0.1]
        
        # 컨텍스트가 없으면 원본 컨텍스트 사용
        if not filtered_contexts and context:
            filtered_contexts = [c[0] for c in scored_contexts[:2]]
        
        # 프롬프트 구성 - 더 상세한 지시사항
        prompt = (
            f"당신은 문서 기반 질의응답 시스템입니다. 아래 정보만을 사용하여 질문에 답변해야 합니다.\n\n"
            f"규칙:\n"
            f"1. 주어진 정보에 명확한 답변이 없다면 '주어진 정보에서 답을 찾을 수 없습니다'라고 솔직히 말해야 합니다.\n"
            f"2. 주어진 정보를 벗어난 내용으로 답변하지 마세요.\n"
            f"3. 답변은 정확하고 완전해야 하며, 한국어로 자연스럽게 작성해야 합니다.\n"
            f"4. 모든 답변은 주어진 정보에 근거해야 합니다.\n\n"
            f"관련 정보:\n"
        )
        
        # 각 컨텍스트 청크 구분하여 추가
        for i, ctx in enumerate(filtered_contexts[:3]):  # 상위 3개만 사용
            prompt += f"[정보 {i+1}]\n{ctx.strip()}\n\n"
        
        prompt += (
            f"질문: {query}\n\n"
            f"답변:"
        )

        try:
            if self.use_huggingface:
                return self._generate_with_huggingface(prompt)
            else:
                return self._generate_with_local_algorithm(prompt)
        except Exception as e:
            logger.error(f"답변 생성 중 오류 발생: {str(e)}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}\n\n대신 직접 관련 정보를 확인해 보세요:\n\n" + "\n".join([f"- {c[:100]}..." for c in filtered_contexts[:2]])
    
    def _generate_with_huggingface(self, prompt: str) -> str:
        logger.info("HuggingFace API로 답변 생성 시도")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 더 나은 생성 파라미터
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,  # 더 긴 응답 허용
                "min_length": 50,       # 최소 길이 설정
                "temperature": 0.7,     # 약간 더 창의적으로
                "top_p": 0.85,          # 더 집중된 샘플링
                "do_sample": True,
                "repetition_penalty": 1.1, # 반복 방지
                "no_repeat_ngram_size": 3, # 3-gram 반복 방지
            }
        }
        
        # 재시도 로직 추가
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    logger.info("HuggingFace API 응답 성공")
                    result = response.json()
                    
                    # 응답 구조에 따라 다른 파싱 처리
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                        
                        # 프롬프트 제거 (API가 전체 프롬프트를 반환하는 경우)
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].strip()
                            
                    elif isinstance(result, dict):
                        generated_text = result.get("generated_text", "")
                    else:
                        generated_text = str(result)
                        
                    # 응답 후처리
                    generated_text = self._postprocess_llm_response(generated_text)
                    
                    # 품질 검사
                    if len(generated_text.strip()) < 20 or not any(char.isalnum() for char in generated_text):
                        logger.warning("생성된 텍스트가 부적절합니다. 재시도합니다.")
                        retry_count += 1
                        continue
                        
                    return generated_text.strip()
                    
                elif response.status_code == 429:  # Rate limit
                    logger.warning("API 속도 제한. 잠시 대기 후 재시도합니다.")
                    retry_count += 1
                    time.sleep(2 ** retry_count)  # 지수 백오프
                    
                elif response.status_code >= 500:  # 서버 오류
                    logger.error(f"서버 오류 ({response.status_code}): {response.text}")
                    retry_count += 1
                    time.sleep(1)
                    
                else:
                    logger.error(f"API 오류 ({response.status_code}): {response.text}")
                    break  # 다른 오류는 재시도하지 않음
                    
            except requests.exceptions.Timeout:
                logger.error("API 요청 시간 초과")
                retry_count += 1
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"API 호출 중 오류: {str(e)}")
                break
                
        # 모든 시도 실패 시 로컬 알고리즘으로 폴백
        logger.info("모든 HuggingFace API 시도 실패. 로컬 알고리즘으로 폴백")
        return self._generate_with_local_algorithm(prompt)

    def _postprocess_llm_response(self, text: str) -> str:
        """LLM 응답 후처리"""
        # 불필요한 접두사 제거
        prefixes_to_remove = [
            "답변:", "answer:", "assistant:", "response:", "chatbot:", "ai:", "챗봇:"
        ]
        
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # 불필요한 따옴표 제거
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
            
        # 줄바꿈 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
       
        return text
class RAGChatbot:

    def __init__(self, llm_api_key: Optional[str] = None, use_huggingface: bool = True, chunk_size: int = 800):
        logger.info("RAG 챗봇 초기화 시작")
        self.document_processor = DocumentProcessor()
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=150)
        self.vector_store = VectorStore()
        self.llm_service = LLMService(api_key=llm_api_key, use_huggingface=use_huggingface)
        # 사용 통계 추적
        self.stats = {
            "processed_documents": 0,
            "processed_chunks": 0,
            "answered_questions": 0,
            "failed_questions": 0
        }
        logger.info("RAG 챗봇 초기화 완료")
    
    async def process_document_async(self, file_path: str) -> int:
        """비동기: 문서 처리 및 인덱싱 - 개선된 방식"""
        try:
            # 문서 읽기
            logger.info(f"문서 처리 시작: {file_path}")
            text = await self.document_processor.read_document_async(file_path)
            
            # 텍스트 전처리 (옵션)
            text = self._preprocess_text(text)
            
            # 청크로 분할
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                logger.warning("문서를 청크로 분할할 수 없습니다.")
                return 0
            
            # 벡터 저장소에 추가
            added = self.vector_store.add_documents(chunks)
            
            # 통계 업데이트
            self.stats["processed_documents"] += 1
            self.stats["processed_chunks"] += added
            
            logger.info(f"문서 처리 완료: {added}개 청크 추가됨")
            return added
        
        except Exception as e:
            logger.error(f"문서 처리 중 오류 발생: {str(e)}")
            raise
    
    def process_document(self, file_path: str) -> int:
        """동기: 문서 처리 및 인덱싱 - 개선된 방식"""
        try:
            # 문서 읽기
            logger.info(f"문서 처리 시작: {file_path}")
            text = self.document_processor.read_document(file_path)
            
            # 텍스트 전처리 (옵션)
            text = self._preprocess_text(text)
            
            # 청크로 분할
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                logger.warning("문서를 청크로 분할할 수 없습니다.")
                return 0
            
            # 벡터 저장소에 추가
            added = self.vector_store.add_documents(chunks)
            
            # 통계 업데이트
            self.stats["processed_documents"] += 1
            self.stats["processed_chunks"] += added
            
            logger.info(f"문서 처리 완료: {added}개 청크 추가됨")
            return added
        
        except Exception as e:
            logger.error(f"문서 처리 중 오류 발생: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """문서 텍스트 전처리"""
        if not text:
            return ""
            
        # 중복 공백 및 불필요한 문자 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 불필요한 특수 문자 제거 (옵션)
        # text = re.sub(r'[^\w\s.,:;?!\'\"()\[\]{}]', '', text)
        
        # 줄바꿈 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    async def answer_question_async(self, query: str) -> str:
        """비동기: 질문에 답변 - 향상된 검색 로직"""
        if not query.strip():
            return "질문을 입력해 주세요."
            
        try:
            # 1. 질문 전처리 및 개선
            processed_query = self._enhance_query(query)
            
            # 2. 관련 문서 검색 - 더 많은 문서 가져오기
            logger.info(f"질문에 대한 관련 문서 검색: {processed_query}")
            relevant_docs = self.vector_store.search(processed_query, top_k=5, threshold=0.6)
            
            if not relevant_docs:
                # 2-1. 첫 검색 실패시 키워드 기반 폴백 검색
                logger.warning("벡터 검색으로 관련 문서를 찾을 수 없어 키워드 검색 시도")
                relevant_docs = self._keyword_search(query, top_k=3)
                
                if not relevant_docs:
                    logger.warning("모든 검색 방법으로 관련 문서를 찾을 수 없습니다.")
                    self.stats["failed_questions"] += 1
                    return "질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 시도하거나 더 많은 문서를 추가해 주세요."
            
            # 3. 검색 결과 후처리 - 중복 제거 및 정렬
            context = self._process_search_results([doc[2] for doc in relevant_docs], query)
            
            # 4. 답변 생성
            answer = await self.llm_service.generate_answer_async(query, context)
            
            # 5. 답변 품질 확인 및 개선
            answer = self._verify_answer_quality(answer, query, context)
            
            # 통계 업데이트
            self.stats["answered_questions"] += 1
            
            logger.info(f"질문 '{query[:30]}...'에 대한 답변 생성 완료")
            return answer
        
        except Exception as e:
            logger.error(f"질문 처리 중 오류: {str(e)}")
            self.stats["failed_questions"] += 1
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def answer_question(self, query: str) -> str:
        """동기: 질문에 답변 - 향상된 검색 로직"""
        if not query.strip():
            return "질문을 입력해 주세요."
            
        try:
            # 1. 질문 전처리 및 개선
            processed_query = self._enhance_query(query)
            
            # 2. 관련 문서 검색 - 더 많은 문서 가져오기
            logger.info(f"질문에 대한 관련 문서 검색: {processed_query}")
            relevant_docs = self.vector_store.search(processed_query, top_k=5, threshold=0.6)
            
            if not relevant_docs:
                # 2-1. 첫 검색 실패시 키워드 기반 폴백 검색
                logger.warning("벡터 검색으로 관련 문서를 찾을 수 없어 키워드 검색 시도")
                relevant_docs = self._keyword_search(query, top_k=3)
                
                if not relevant_docs:
                    logger.warning("모든 검색 방법으로 관련 문서를 찾을 수 없습니다.")
                    self.stats["failed_questions"] += 1
                    return "질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 시도하거나 더 많은 문서를 추가해 주세요."
            
            # 3. 검색 결과 후처리 - 중복 제거 및 정렬
            context = self._process_search_results([doc[2] for doc in relevant_docs], query)
            
            # 4. 답변 생성
            answer = self.llm_service.generate_answer(query, context)
            
            # 5. 답변 품질 확인 및 개선
            answer = self._verify_answer_quality(answer, query, context)
            
            # 통계 업데이트
            self.stats["answered_questions"] += 1
            
            logger.info(f"질문 '{query[:30]}...'에 대한 답변 생성 완료")
            return answer
        
        except Exception as e:
            logger.error(f"질문 처리 중 오류: {str(e)}")
            self.stats["failed_questions"] += 1
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _enhance_query(self, query: str) -> str:
        """질문 개선: 노이즈 제거 및 더 명확한 검색어 생성"""
        # 기본 전처리
        processed = re.sub(r'[^\w\s\?\.]', ' ', query)
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # 한국어 특화 처리 (조사 처리 등)
        korean_postpositions = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로']
        for pp in korean_postpositions:
            processed = re.sub(r'(\w+)' + pp + r'\b', r'\1', processed)
        
        return processed
    
    def _keyword_search(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """키워드 기반 검색 (벡터 검색 실패시 폴백)"""
        if not self.vector_store.documents:
            return []
        
        # 질문에서 키워드 추출
        try:
            words = re.findall(r'\w+', query.lower())
            # 불용어 제거
            korean_stopwords = ['그', '이', '저', '것', '이것', '저것', '그것', '무엇', '어떤', '때', '수', '등', '들', '더', '나', '너', '우리', '저희', '당신', '그들']
            keywords = [w for w in words if len(w) > 1 and w not in korean_stopwords]
            
            if not keywords:
                return []
                
            # 문서에서 키워드 매칭 점수 계산
            scored_docs = []
            for idx, doc in enumerate(self.vector_store.documents):
                score = 0
                for keyword in keywords:
                    if keyword in doc.lower():
                        score += 1
                if score > 0:
                    scored_docs.append((idx, score / len(keywords), doc))
            
            # 점수 기준 정렬 및 상위 k개 반환
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:top_k]
        except Exception as e:
            logger.error(f"키워드 검색 중 오류: {str(e)}")
            return []
    
    def _process_search_results(self, docs: List[str], query: str) -> List[str]:
        """검색 결과 후처리: 중복 제거 및 가장 관련 높은 순으로 재정렬"""
        if not docs:
            return []
        
        # 간단한 중복 문장 제거 (85% 이상 유사하면 중복으로 간주)
        unique_docs = []
        for doc in docs:
            is_duplicate = False
            for existing in unique_docs:
                # 자카드 유사도 계산
                set1 = set(doc.split())
                set2 = set(existing.split())
                jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                
                if jaccard > 0.85:  # 85% 이상 중복
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_docs.append(doc)
        
        return unique_docs
    
    def _verify_answer_quality(self, answer: str, query: str, context: List[str]) -> str:
        """답변 품질 확인 및 개선"""
        # 답변이 너무 짧은지 확인
        if len(answer.split()) < 5:
            logger.warning("답변이 너무 짧습니다.")
            return f"답변이 불충분합니다. 제공된 정보를 바탕으로 다음 내용을 참고하세요:\n\n{' '.join(context[:1])}"
        
        # "모르겠다" 유형의 응답 감지 및 개선
        uncertainty_phrases = ['잘 모르', '알 수 없', '정보가 없', '확인할 수 없', '주어진 정보에 없']
        for phrase in uncertainty_phrases:
            if phrase in answer:
                logger.info("불확실한 답변 감지, 컨텍스트 직접 제공")
                if len(context) > 0:
                    return f"{answer}\n\n참고할 수 있는 관련 정보:\n{context[0]}"
        
        return answer
    
    def get_statistics(self) -> Dict[str, Any]:
        """챗봇 상태 및 통계 정보 반환 - 개선된 통계"""
        try:
            vector_count = len(self.vector_store.vectors)
            vector_dim = len(self.vector_store.vectors[0]) if vector_count > 0 else 0
            memory_usage = sum(v.nbytes for v in self.vector_store.vectors) / (1024 * 1024) if vector_count > 0 else 0
            
            return {
                "document_chunks": vector_count,
                "vector_dimension": vector_dim,
                "memory_usage_mb": round(memory_usage, 2),
                "model_name": self.vector_store.model_name,
                "llm_type": "HuggingFace API" if self.llm_service.use_huggingface else "로컬 알고리즘",
                "device": str(self.vector_store.device),
                "processed_documents": self.stats["processed_documents"],
                "processed_chunks": self.stats["processed_chunks"],
                "answered_questions": self.stats["answered_questions"],
                "failed_questions": self.stats["failed_questions"],
                "success_rate": round(self.stats["answered_questions"] / max(1, self.stats["answered_questions"] + self.stats["failed_questions"]) * 100, 1)
            }
        except Exception as e:
            logger.error(f"통계 정보 생성 중 오류: {str(e)}")
            return {"error": str(e)}
    
    def clear(self) -> bool:
        """모든 문서 및 벡터 데이터 삭제"""
        try:
            # 벡터 및 문서 초기화
            self.vector_store.vectors = []
            self.vector_store.documents = []
            
            # 통계 초기화
            self.stats = {
                "processed_documents": 0,
                "processed_chunks": 0, 
                "answered_questions": 0,
                "failed_questions": 0
            }
            
            # 메모리 정리
            import gc
            gc.collect()
            
            # PyTorch 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("챗봇 데이터 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"데이터 초기화 중 오류: {str(e)}")
            return False