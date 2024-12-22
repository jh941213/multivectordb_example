from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain_core.runnables import ConfigurableField
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="KT DS QA System API")

class Question(BaseModel):
    query: str
    category: str

class Answer(BaseModel):
    question: str
    category: str
    sources: Optional[List[str]] = None
    error: Optional[str] = None

class MultiVectorDBSearch:
    def __init__(self, base_path="/app"):
        """QA 시스템 초기화"""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.base_path = base_path
        
        # 각 카테고리별 벡터 DB 경로 설정
        self.db_paths = {
            "COMPANY": os.path.join(base_path, "company"),
            "HR": os.path.join(base_path, "hr"),
            "Product": os.path.join(base_path, "product"),
            "Finance": os.path.join(base_path, "finance")
        }
        
        # 저장된 벡터 DB와 검색기 초기화
        self.ensemble_retrievers = {}
        
        for category, db_path in self.db_paths.items():
            try:
                if os.path.exists(db_path):
                    # FAISS 벡터 DB 로드
                    vector_store = FAISS.load_local(
                        db_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # FAISS Retriever 설정
                    faiss_retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 2,
                            "fetch_k": 4,
                            "lambda_mult": 0.7
                        }
                    )
                    
                    # 문서 내용 추출하여 BM25 검색기 초기화
                    docs = vector_store.similarity_search("", k=2)
                    texts = [doc.page_content for doc in docs]
                    bm25_retriever = KiwiBM25Retriever.from_texts(texts)
                    bm25_retriever.k = 2
                    
                    # Ensemble Retriever 생성
                    self.ensemble_retrievers[category] = EnsembleRetriever(
                        retrievers=[bm25_retriever, faiss_retriever],
                    ).configurable_fields(
                        weights=ConfigurableField(
                            id="ensemble_weights",
                            name="Ensemble Weights",
                            description="Weights for BM25 and FAISS retrievers"
                        )
                    )
                    
                    print(f"{category} Ensemble Retriever 초기화 완료: {db_path}")
                else:
                    print(f"{category} 벡터 DB 경로가 존재하지 않습니다: {db_path}")
            except Exception as e:
                print(f"{category} 초기화 실패: {e}")

    def get_relevant_docs(self, query: str, category: str) -> Dict:
        try:
            if category not in self.ensemble_retrievers:
                raise HTTPException(status_code=404, detail=f"{category} 카테고리의 검색기를 찾을 수 없습니다.")
            
            # Ensemble 검색 실행 (BM25: 0.9, FAISS: 0.1 가중치)
            config = {"configurable": {"ensemble_weights": [0.9, 0.1]}}
            docs = self.ensemble_retrievers[category].invoke(
                query,
                config=config
            )
            
            # 결과 포맷팅
            return {
                "question": query,
                "category": category,
                "sources": [doc.page_content for doc in docs]
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

qa_system = MultiVectorDBSearch()

@app.post("/api/v1/search", response_model=Answer)
async def search_documents(question: Question):
    """질문과 카테고리를 받아 관련 문서를 검색합니다."""
    try:
        result = qa_system.get_relevant_docs(question.query, question.category)
        return {
            "question": result["question"],
            "category": result["category"],
            "sources": result.get("sources", []),
            "error": None
        }
    except Exception as e:
        return {
            "question": question.query,
            "category": question.category,
            "sources": None,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """서버 상태를 확인합니다."""
    return {"status": "healthy"}