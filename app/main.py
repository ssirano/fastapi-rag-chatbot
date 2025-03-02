from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import RedirectResponse  
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import tempfile
import uuid
import time
import shutil
import asyncio
import aiofiles
import logging

# RAG 모듈 임포트
from .rag_chatbot import RAGChatbot

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG 문서 분석 API",
    description="문서를 분석하고 질문에 답변하는 RAG 기반 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드에서 API 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 특정 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임시 저장소 (실제 애플리케이션에서는 데이터베이스 사용 권장)
CHATBOT_INSTANCES = {}
UPLOAD_FOLDER = "temp_uploads"

# 업로드 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 요청/응답 모델 정의
class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str

class ProcessStatusResponse(BaseModel):
    status: str
    message: str
    chunks: Optional[int] = None

# 세션 정리 함수
async def cleanup_session(session_id: str, delay: int = 3600):
    """일정 시간 후 세션 리소스 정리"""
    await asyncio.sleep(delay)  # 1시간
    if session_id in CHATBOT_INSTANCES:
        del CHATBOT_INSTANCES[session_id]
        logger.info(f"세션 {session_id} 정리됨")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 루트 경로에서 index.html로 리디렉션
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# 엔드포인트 정의
@app.post("/api/upload", response_model=ProcessStatusResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_huggingface: bool = Form(True),
    api_key: Optional[str] = Form(None)
):
    """문서 업로드 및 처리 엔드포인트"""
    try:
        # 세션 ID 생성
        session_id = str(uuid.uuid4())
        
        # 파일 임시 저장
        file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{file.filename}")
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # RAG 챗봇 인스턴스 생성 및 문서 처리
        chatbot = RAGChatbot(llm_api_key=api_key, use_huggingface=use_huggingface)
        num_chunks = chatbot.process_document(file_path)
        
        # 인스턴스 저장
        CHATBOT_INSTANCES[session_id] = chatbot
        
        # 세션 자동 정리 태스크 (1시간 후)
        background_tasks.add_task(cleanup_session, session_id)
        
        # 임시 파일 삭제
        os.remove(file_path)
        
        return {
            "status": "success",
            "message": f"문서 처리 완료. 세션 ID: {session_id}",
            "chunks": num_chunks
        }
    
    except Exception as e:
        logger.error(f"문서 업로드 중 오류: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """질문에 답변하는 엔드포인트"""
    session_id = request.session_id
    query = request.query
    
    if session_id not in CHATBOT_INSTANCES:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다. 문서를 다시 업로드하세요.")
    
    try:
        chatbot = CHATBOT_INSTANCES[session_id]
        answer = chatbot.answer_question(query)
        return {"answer": answer}
    
    except Exception as e:
        logger.error(f"답변 생성 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """세션 상태 확인 엔드포인트"""
    if session_id in CHATBOT_INSTANCES:
        return {"status": "active", "message": "세션이 활성화되어 있습니다."}
    else:
        return {"status": "inactive", "message": "세션을 찾을 수 없습니다."}

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제 엔드포인트"""
    if session_id in CHATBOT_INSTANCES:
        del CHATBOT_INSTANCES[session_id]
        return {"status": "success", "message": "세션이 삭제되었습니다."}
    else:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

@app.get("/")
async def root():
    """API 상태 확인 엔드포인트"""
    return {"status": "OK", "message": "RAG 문서 분석 API가 실행 중입니다."}

# 서버 시작 시 임시 폴더 정리
@app.on_event("startup")
async def startup_event():
    # 임시 폴더 비우기
    for file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"시작 시 임시 파일 삭제 오류: {str(e)}")

# 서버 종료 시 정리
@app.on_event("shutdown")
async def shutdown_event():
    # 모든 리소스 정리
    CHATBOT_INSTANCES.clear()
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)

# 직접 실행 시
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)