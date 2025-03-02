import uvicorn

if __name__ == "__main__":

    print("FastAPI 서버 시작 중...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True, log_level="info")