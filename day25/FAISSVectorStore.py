import faiss
import numpy as np
import os
from dotenv import load_dotenv
from OpenAiembedding import OpenAIEmbedder
load_dotenv()

class FAISSVectorStore:
    def __init__(self, dimension):
        self.dimension = dimension # 모든 벡터가 같은 차원을 가져야 FAISS 인덱스에 저장 가능
        self.index = faiss.IndexFlatIP(dimension) # Inner Product 인덱스 생성
        self.documents = []
        self.document_ids = []

    def add_documents(self, documents, embeddings, document_ids=None):
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # 임베딩 정규화
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        if document_ids is None:
            document_ids = [f"doc_{len(self.documents) - len(documents) + i}"
                            for i in range(len(documents))]
        self.document_ids.extend(document_ids)
        print(f"{len(documents)}")

    def search(self, query_embedding, top_k=5):
        query_embedding = query_embedding / np.linalg.norm(query_embedding) # 쿼리 임배딩 정규화
        query_embedding = query_embedding.reshape(1, -1).astype('float32') # FAISS 검색을 위한 형태 변화
        scores, indicies = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indicies[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'document_id': self.document_ids[idx],
                    'index': int(idx)
                })
        return results
    
sample_docs = [
    "딥러닝은 인공신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 머신러닝 기법입니다.",
    "머신러닝은 데이터로부터 패턴을 학습하여 예측이나 분류를 수행하는 인공지능 기술입니다.",
    "자연어처리는 컴퓨터가 인간의 언어를 이해하고 처리할 수 있게 하는 기술 분야입니다.",
    "컴퓨터 비전은 컴퓨터가 이미지나 비디올르 해석하고 이해할 수 있게 하는 기술입니다.",
    "강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 방향으로 학습하는 방법입니다."
]

api_key = os.getenv("OPENAI_API_KEY")
embedder = OpenAIEmbedder(api_key)
doc_embeddings_list = embedder.get_batch_embeddings(sample_docs)

if doc_embeddings_list:
    doc_embeddings = np.array(doc_embeddings_list)
    faiss_store = FAISSVectorStore(dimension=doc_embeddings_list.shape[1])
    faiss_store.add_documents(sample_docs, doc_embeddings)

query_embedding_list = embedder.get_embedding("딥러닝 학습 방법")

if query_embedding_list:
    query_embedding = np.array(query_embedding_list)
    fast_results = faiss_store.search(query_embedding, top_k=3)


