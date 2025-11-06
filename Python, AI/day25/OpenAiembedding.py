from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
load_dotenv()

class OpenAIEmbedder:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"
    
    # 텍스트의 임베딩 벡터 가져오기
    def get_embedding(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    # 여러 텍스트의 임베딩을 한번에 가져오기
    def get_batch_embeddings(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]
    
embedder = OpenAIEmbedder(os.getenv("OPENAI_API_KEY"))
embedding = embedder.get_embedding("안녕하세요")
print(f"임베딩 차원: {len(embedding) if embedding else 0}")
print(f"임베딩 벡터(처음 10개 값): {embedding[:10]}")
similarity = cosine_similarity([embedding], [embedding])
print(f"유사도: {similarity[0][0]}")