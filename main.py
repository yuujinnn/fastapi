from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import faiss
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# FastAPI 앱 생성
app = FastAPI()

# 정적 파일 경로 설정
app.mount("/static", StaticFiles(directory="ai/dist"), name="static-ai")
app.mount("/wedding/static", StaticFiles(directory="wedding/dist"), name="static-wedding")
app.mount("/wedding/images", StaticFiles(directory="wedding/images"), name="wedding-images")


# 메인 페이지
@app.get("/", response_class=HTMLResponse)
async def read_ai():
    with open("ai/dist/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# 웨딩 페이지
@app.get("/wedding", response_class=HTMLResponse)
async def read_wedding():
    with open("wedding/dist/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# 데이터 로드 및 전처리
file_path = "웨딩전체_가공완료_정규화VER.csv"
wedding_data = pd.read_csv(file_path)

# 결합된 텍스트 컬럼 생성
wedding_data['combined_text'] = wedding_data[['색감', '분위기', '고유한 특징', '촬영 구도', '동작', '의상']].apply(
    lambda x: ' '.join(x.dropna()), axis=1
)

# 모델 로드
sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
kobart = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')

# 임베딩 생성 및 추가
wedding_data['embeddings'] = list(sentence_model.encode(wedding_data['combined_text'].tolist()))
embeddings = np.array(wedding_data['embeddings'].tolist())

# FAISS 인덱스 초기화 및 데이터 추가
dimension = embeddings.shape[1]  # 임베딩 벡터 차원
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 입력 데이터 모델 정의
class UserInput(BaseModel):
    text: Optional[str] = None
    options: Optional[dict] = None  # 색감, 분위기, 고유한 특징 등 선택지를 딕셔너리로 전달
    top_k: int = 3

# KoBART 기반 키워드 추출 함수
def extract_keywords_from_text(user_input):
    prompt = f"""
    아래 문장에서 주요 키워드(색감, 분위기, 동작, 배경)를 추출하세요:
    입력: {user_input}
    출력: 색감: <색감>, 분위기: <분위기>, 동작: <동작>, 배경: <배경>."""
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).input_ids
    summary_ids = kobart.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

# 추천 함수 (공통)
def recommend_photographers(user_embedding, top_k=3, clothing_filter=None):
    # '의상' 필터 적용
    if clothing_filter:
        filtered_data = wedding_data[wedding_data['의상'].str.contains(clothing_filter, na=False)]
        if filtered_data.empty:
            raise HTTPException(status_code=404, detail="해당 의상에 대한 데이터가 없습니다.")
    else:
        filtered_data = wedding_data

    # 필터링된 데이터에 대해 FAISS 인덱스를 생성
    filtered_embeddings = np.array(filtered_data['embeddings'].tolist())
    dimension = filtered_embeddings.shape[1]  # 임베딩 차원
    filtered_index = faiss.IndexFlatL2(dimension)
    filtered_index.add(filtered_embeddings)

    # 유사도 검색 수행
    distances, indices = filtered_index.search(np.array(user_embedding), top_k)
    recommendations = filtered_data.iloc[indices[0]]

    # 결과 정리 및 반환
    recommendations['filtered_image_filename'] = recommendations['image_filename'].str.replace(r'_\d{2}\.(jpg|png)', '', regex=True)
    return recommendations[['filtered_image_filename', 'image_filename', '색감', '분위기', '고유한 특징', '촬영 구도', '동작', '의상']]


# FastAPI 엔드포인트
@app.post("/wedding")
async def recommend(user_input: UserInput):
    try:
        clothing_filter = user_input.options.get('의상') if user_input.options else None
        text_embedding = None
        options_embedding = None

        # text 입력 처리
        if user_input.text:
            keywords = extract_keywords_from_text(user_input.text)
            text_embedding = sentence_model.encode([keywords])

        # options 입력 처리
        if user_input.options:
            combined_options = ' '.join([f"{k}: {v}" for k, v in user_input.options.items() if v])
            options_embedding = sentence_model.encode([combined_options])

        # 입력값이 없을 경우 에러 처리
        if text_embedding is None and options_embedding is None:
            raise HTTPException(status_code=400, detail="입력값(text 또는 options)이 필요합니다.")

        # 유사도 평균 계산
        if text_embedding is not None and options_embedding is not None:
            # text와 options 둘 다 있는 경우
            combined_embedding = (text_embedding + options_embedding) / 2
        else:
            # 하나만 있는 경우
            combined_embedding = text_embedding if text_embedding is not None else options_embedding

        # 추천 실행 (의상 필터 추가)
        recommendations = recommend_photographers(combined_embedding, user_input.top_k, clothing_filter)
        return recommendations.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
