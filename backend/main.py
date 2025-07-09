from fastapi import FastAPI, Depends, HTTPException, Form, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware
from passlib.context import CryptContext
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from openai import OpenAI
import os
from dotenv import load_dotenv
from models import User
from db import SessionLocal
from models import LikedText
import asyncio
import time  
import json
from typing import Optional
from datetime import datetime, timedelta
import jwt

# Загрузка переменных окружения
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("❌ API-ключ не найден! Проверь .env файл!")

print(f"✅ API-ключ загружен: {openai_key[:5]}...")

# Создаем ОДИН экземпляр FastAPI
app = FastAPI(
    title="Рэп Генератор API",
    description="API для генерации текстов рэп-песен в стиле Славы КПСС с использованием GPT-4",
    version="1.0.0",
    docs_url="/docs",  
    redoc_url="/redoc", 
    openapi_tags=[
        {
            "name": "Генерация текстов",
            "description": "Операции, связанные с генерацией текстов рэпа",
        },
        {
            "name": "Аутентификация",
            "description": "Операции для регистрации, входа и выхода пользователей",
        },
        {
            "name": "Избранное",
            "description": "Операции для работы с избранными текстами",
        },
    ] 
)

# Добавляем middleware в правильном порядке
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Сессии cookies - ВАЖНО: добавляем к единственному экземпляру app
app.add_middleware(SessionMiddleware, secret_key="supersecretkey")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Загрузка песен
INDEX_PATH = "faiss_index"
with open("all_songs.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

songs = [song.strip() for song in raw_text.split("\n\n") if song.strip()]
print(f"🎵 Загружено {len(songs)} песен!")

client = OpenAI(
    api_key=openai_key,
    base_url="https://api.vsegpt.ru/v1",
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Конфигурация токенов
SECRET_KEY = "Very very secret key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Модели данных
class TokenData(BaseModel):
    username: Optional[str] = None
    token_type: str

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str

class RequestModel(BaseModel):
    prompt: str

class LikeTextRequest(BaseModel):
    text: str

def load_or_create_index(texts, index_path):
    if os.path.exists(index_path):
        print("🔄 Загружаем сохранённый FAISS-индекс...")
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
            openai_api_base="https://api.vsegpt.ru/v1",
        )
        db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("🚀 Создаём новый FAISS-индекс...")
        splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', ' '], chunk_size=128, chunk_overlap=64)
        source_chunks = [
            Document(page_content=chunk, metadata={'chunkID': i})
            for i, chunk in enumerate(splitter.split_text("\n\n".join(texts)))
        ]
        if source_chunks:
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=openai_key,
                openai_api_base="https://api.vsegpt.ru/v1",
            )
            db = FAISS.from_documents(source_chunks, embedding_model)
            db.save_local(index_path)
            print("FAISS-индекс сохранён!")
    return db

knowledge_base_index = load_or_create_index(songs, INDEX_PATH)

def create_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Функция генерации текста
def generate_song(prompt: str):
    try:
        docs = knowledge_base_index.similarity_search(prompt, k=3) if knowledge_base_index else []
        similar_texts = [doc.page_content for doc in docs] if docs else ["База пуста, но ты можешь написать свой куплет!"]

        context = "\n".join(similar_texts)
        messages = [
            {"role": "system", "content": "Ты рэпер Слава КПСС. Пиши реп в его стиле, важно чтобы были рифмы, т.е не было строчек без рифм"},
            {"role": "user", "content": f"Вот примеры строк:\n{context}\n\nТеперь продолжи: {prompt}"}
        ]

        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.8
        )

        return completion.choices[0].message.content
    except Exception as e:
        print(f"Ошибка генерации: {e}")
        return "Ошибка при генерации текста. Попробуйте позже."

# Маршруты
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/token", response_model=Token, tags=["Аутентификация"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Получение пары токенов доступа для авторизации (для использования в свагере).
    """
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.username == form_data.username).first()
        
        if not user or not pwd_context.verify(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=401,
                detail="Неверное имя пользователя или пароль",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_token(
            data={"sub": user.username, "type": "access"}, 
            expires_delta=access_token_expires
        )
        
        # refresh token
        refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        refresh_token = create_token(
            data={"sub": user.username, "type": "refresh"}, 
            expires_delta=refresh_token_expires
        )
        
        db.close()
        
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "refresh_token": refresh_token
        }
    except Exception as e:
        print(f"Ошибка в токен-эндпоинте: {str(e)}")
        raise

@app.post("/refresh", response_model=Token, tags=["Аутентификация"])
async def refresh_access_token(refresh_token: str):
    """
    Обновление access token с помощью refresh token.
    """
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        token_type = payload.get("type")
        
        if username is None or token_type != "refresh":
            raise HTTPException(status_code=401, detail="Недействительный refresh token")
        
        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        db.close()
        
        if not user:
            raise HTTPException(status_code=401, detail="Пользователь не найден")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_token(
            data={"sub": user.username, "type": "access"}, 
            expires_delta=access_token_expires
        )
        
        refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        new_refresh_token = create_token(
            data={"sub": user.username, "type": "refresh"}, 
            expires_delta=refresh_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "refresh_token": new_refresh_token
        }
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Недействительный refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    """
    Комбинированная аутентификация, работающая как с сессионными куками, так и с OAuth2 токенами.
    """
    user = request.session.get("user")
    
    if not user and token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            token_type = payload.get("type")
            
            if username is None or token_type != "access":
                raise HTTPException(status_code=401, detail="Недействительный токен")
            
            db = SessionLocal()
            user = db.query(User).filter(User.username == username).first()
            db.close()
            
            if not user:
                raise HTTPException(status_code=401, detail="Пользователь не найден")
            
            return user.username
            
        except Exception as e: 
            print(f"Ошибка JWT: {str(e)}")
            pass
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Не авторизован",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

@app.post("/generate", response_model=dict, tags=["Генерация текстов"])
async def generate(request: RequestModel, user: str = Depends(get_current_user)):
    """
    Генерирует текст рэп-песни на основе введенного промпта.
    
    - **prompt**: Текст или тема для генерации
    
    Требуется авторизация.
    """
    result = generate_song(request.prompt)
    return {"lyrics": result}

@app.post("/register", tags=["Аутентификация"]) 
async def register(username: str = Form(...), password: str = Form(...)):
    """
    Регистрация нового пользователя.
    
    - **username**: Имя пользователя
    - **password**: Пароль
    """
    db = SessionLocal()
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        db.close()
        raise HTTPException(status_code=400, detail="❌ Пользователь уже существует!")

    hashed_pw = pwd_context.hash(password)
    new_user = User(username=username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    db.close()

    return JSONResponse(content={"message": "✅ Регистрация успешна!"})

@app.post("/login", tags=["Аутентификация"])
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """
    Вход пользователя в систему.
    
    - **username**: Имя пользователя
    - **password**: Пароль
    """
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()

    if not user or not pwd_context.verify(password, user.hashed_password):
        return JSONResponse(
            status_code=401,
            content={"message": "❌ Неверное имя пользователя или пароль!"}
        )

    request.session["user"] = username
    return JSONResponse(content={"message": "✅ Вход выполнен!"})

@app.post("/logout", tags=["Аутентификация"])
async def logout(request: Request):
    """
    Выход пользователя из системы.
    """
    request.session.clear()
    return JSONResponse(content={"message": "👋 Выход выполнен!"})

@app.get("/protected")
async def protected(user: str = Depends(get_current_user)):
    return {"message": f"🔐 Доступ разрешён, {user}!"}

@app.post("/like", tags=["Избранное"])
async def like_text(request: LikeTextRequest, user: str = Depends(get_current_user)):
    """
    Сохранение текста в избранное.
    
    - **text**: Текст для сохранения
    
    Требуется авторизация.
    """
    db = SessionLocal()
    db_user = db.query(User).filter(User.username == user).first()
    if not db_user:
        db.close()
        raise HTTPException(status_code=404, detail="Пользователь не найден!")

    liked = LikedText(user_id=db_user.id, text_content=request.text)
    db.add(liked)
    db.commit()
    db.refresh(liked)
    db.close()

    return JSONResponse(content={"message": "✅ Текст сохранён в избранное!"})

@app.get("/liked", tags=["Избранное"])
async def get_liked_texts(user: str = Depends(get_current_user)):
    """
    Получение всех текстов, сохраненных в избранное пользователем.
    
    Требуется авторизация.
    """
    db = SessionLocal()
    db_user = db.query(User).filter(User.username == user).first()
    if not db_user:
        db.close()
        raise HTTPException(status_code=404, detail="Пользователь не найден!")

    texts = db.query(LikedText).filter(LikedText.user_id == db_user.id).all()
    db.close()

    return {"liked_texts": [{"id": t.id, "text_content": t.text_content, "created_at": t.created_at} for t in texts]}

@app.get("/generator")
async def generator_page(user: str = Depends(get_current_user)):
    filepath = "static/generator.html"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="generator.html не найден!")
    return FileResponse(filepath)

@app.get("/generate/stream", tags=["Генерация текстов"])
async def stream_generation(request: Request, prompt: str = "Без названия"):
    """
    Потоковая генерация текста рэп-песни с отображением этапов процесса.
    
    - **prompt**: Текст или тема для генерации
    
    Возвращает Server-Sent Events (SSE) с прогрессом и финальным результатом.
    """
    async def event_generator():
        steps = [
            "🔄 Генерируем...",
            "🎵 Нашли похожие строки...",
            "🧠 Обращаемся к OpenAI...",
            "Еще несколько секунд...",
        ]

        for step in steps:
            if await request.is_disconnected():
                print("Клиент отключился")
                break
            yield f"data: {step}\n\n"
            await asyncio.sleep(2)

        try:
            lyrics = generate_song(prompt)
            yield f"data: ✅ Готово!\n\n"
            await asyncio.sleep(0.5)
            
            payload = json.dumps({"lyrics": lyrics})
            yield f"data: {payload}\n\n"
        except Exception as e:
            print(f"❌ Ошибка при генерации: {e}")
            error_payload = json.dumps({"error": "Произошла ошибка при генерации текста"})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/favorites")
async def favorites_page(user: str = Depends(get_current_user)):
    """
    Страница с избранными текстами пользователя.
    Требуется авторизация.
    """
    filepath = "static/favorites.html"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="favorites.html не найден!")
    return FileResponse(filepath)