from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import requests
from fastapi import UploadFile, File
import os
from pypdf import PdfReader
import chromadb

app = FastAPI(title="AI学习辅导Agent")

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 大模型配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "你的API_KEY"
MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
# Chroma向量数据库
CHROMA_DIR = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="study_materials")

# Embedding配置
EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# 数据库配置
DB_URL = "mysql+pymysql://root:你的密码@127.0.0.1:3306/study_agent"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# ===== 学习资料表 =====
class Material(Base):
    __tablename__ = "material"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(200), comment="资料标题")
    file_name = Column(String(200), comment="文件名")
    create_time = Column(DateTime, default=datetime.now, comment="上传时间")


# ===== 题目表 =====
class Question(Base):
    __tablename__ = "question"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    material_id = Column(Integer, comment="关联哪份资料")
    question_type = Column(String(20), comment="题型：choice / essay")
    question = Column(Text, comment="题目内容")
    options = Column(Text, comment="选项（JSON字符串）")
    answer = Column(Text, comment="正确答案")
    explanation = Column(Text, comment="解析")
    create_time = Column(DateTime, default=datetime.now)


# ===== 答题记录表 =====
class AnswerRecord(Base):
    __tablename__ = "answer_record"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question_id = Column(Integer, comment="关联哪道题")
    user_answer = Column(Text, comment="用户答案")
    is_correct = Column(Integer, comment="是否正确：1对 0错")
    submit_time = Column(DateTime, default=datetime.now)


# 自动建表
Base.metadata.create_all(bind=engine)
def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return chunks

def get_embedding(text: str):
    resp = requests.post(
        EMBEDDING_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60
    )
    return resp.json()["data"][0]["embedding"]

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise Exception("不支持的文件格式，仅支持PDF、TXT、MD")
# ===== 上传资料接口 =====
@app.post("/material/upload", summary="上传学习资料")
def upload_material(
    file: UploadFile = File(...),
    title: str = "",
    db = Depends(get_db)
):
    # 1. 确保uploads文件夹存在
    os.makedirs("uploads", exist_ok=True)

    # 2. 保存文件到本地
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # 3. 把文件信息存进MySQL
    material = Material(
        title=title or file.filename,
        file_name=file.filename
    )
    db.add(material)
    db.commit()
    db.refresh(material)

    # 4. 读取文件内容，分块，存进Chroma
    text = load_document(file_path)
    chunks = split_text(text, chunk_size=300, overlap=50)

    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[f"material_{material.id}_chunk_{i}"],
            metadatas=[{"material_id": material.id}]
        )

    return {
        "code": 200,
        "message": f"上传成功，共存入 {len(chunks)} 个文本块",
        "material_id": material.id
    }
# ===== 出题接口 =====
from pydantic import BaseModel
import json

class GenerateRequest(BaseModel):
    material_id: int
    question_count: int = 5

@app.post("/quiz/generate", summary="根据资料出题")
def generate_quiz(
    request: GenerateRequest,
    db = Depends(get_db)
):
    # 1. 从Chroma检索这份资料的所有文本块
    results = collection.get(
        where={"material_id": request.material_id}
    )

    # 2. 拼接所有资料内容
    context = "\n\n".join(results["documents"])
    print("查到的文本块数量：", len(results["documents"]))
    print("context内容：", repr(context[:200]))

    # 3. 调大模型出题
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": f"""你是一个出题专家。请根据以下学习资料，出{request.question_count}道选择题。

要求：
1. 每道题4个选项，只有一个正确答案
2. 必须严格基于资料内容出题
3. 每道题要有解析

重要：只返回JSON，不要有任何其他文字、解释或markdown标记。

JSON格式：
{{"questions":[{{"question":"题目","options":{{"A":"选项A","B":"选项B","C":"选项C","D":"选项D"}},"answer":"A","explanation":"解析"}}]}}

学习资料：
{context}"""

            }
        ],
        "temperature": 0.3
    }

    resp = requests.post(url=API_URL, headers=headers, json=payload, timeout=120)
    result = resp.json()
    content = result["choices"][0]["message"]["content"]
    print("模型返回的原始内容：", repr(content))

    # 4. 解析模型返回的JSON
    # 去掉可能的markdown代码块标记
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    # 找到第一个{和最后一个}之间的内容
    start = content.find("{")
    end = content.rfind("}")
    content = content[start:end + 1]
    quiz_data = json.loads(content)

    # 5. 把题目存进MySQL
    questions = []
    for q in quiz_data["questions"]:
        question = Question(
            material_id=request.material_id,
            question_type="choice",
            question=q["question"],
            options=json.dumps(q["options"], ensure_ascii=False),
            answer=q["answer"],
            explanation=q["explanation"]
        )
        db.add(question)
        db.commit()  # 先提交，拿到id
        db.refresh(question)
        q["id"] = question.id  # 把数据库生成的id加到返回里
        questions.append(q)

    db.commit()

    return {
        "code": 200,
        "message": f"出题成功，共{len(questions)}道题",
        "questions": questions
    }
# ===== 提交答案接口 =====
class AnswerRequest(BaseModel):
    question_id: int
    user_answer: str

@app.post("/quiz/submit", summary="提交答案并批改")
def submit_answer(
    request: AnswerRequest,
    db = Depends(get_db)
):
    # 1. 从MySQL查出这道题
    question = db.query(Question).filter(Question.id == request.question_id).first()

    if not question:
        return {"code": 404, "message": "题目不存在"}

    # 2. 对比答案
    is_correct = 1 if request.user_answer == question.answer else 0

    # 3. 存答题记录
    record = AnswerRecord(
        question_id=request.question_id,
        user_answer=request.user_answer,
        is_correct=is_correct
    )
    db.add(record)
    db.commit()

    # 4. 返回批改结果
    return {
        "code": 200,
        "data": {
            "question": question.question,
            "your_answer": request.user_answer,
            "correct_answer": question.answer,
            "result": "正确" if is_correct else "错误",
            "explanation": question.explanation
        }
    }
# ===== 错题本接口 =====
@app.get("/quiz/wrong", summary="查看错题本")
def get_wrong_questions(
    db = Depends(get_db)
):
    # 1. 查出所有答错的答题记录
    wrong_records = db.query(AnswerRecord).filter(
        AnswerRecord.is_correct == 0
    ).all()

    # 2. 查出每道错题的详细信息
    wrong_list = []
    for record in wrong_records:
        question = db.query(Question).filter(Question.id == record.question_id).first()
        if question:
            wrong_list.append({
                "question_id": question.id,
                "question": question.question,
                "options": question.options,
                "your_answer": record.user_answer,
                "correct_answer": question.answer,
                "explanation": question.explanation
            })

    return {
        "code": 200,
        "total": len(wrong_list),
        "wrong_questions": wrong_list
    }
# ===== 资料列表接口 =====
@app.get("/material/list", summary="查看所有上传的资料")
def get_material_list(
    db = Depends(get_db)
):
    materials = db.query(Material).order_by(Material.id.desc()).all()

    result = []
    for m in materials:
        result.append({
            "id": m.id,
            "title": m.title,
            "file_name": m.file_name,
            "create_time": str(m.create_time)
        })

    return {
        "code": 200,
        "total": len(result),
        "materials": result
    }
# ===== 答题统计接口 =====
@app.get("/quiz/stats", summary="答题统计")
def get_stats(
    db = Depends(get_db)
):
    total = db.query(AnswerRecord).count()
    correct = db.query(AnswerRecord).filter(AnswerRecord.is_correct == 1).count()
    wrong = total - correct
    accuracy = round(correct / total * 100, 1) if total > 0 else 0

    return {
        "code": 200,
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": f"{accuracy}%"
    }
