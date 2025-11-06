import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI 실전 프로젝트"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    API_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    DATABASE_URL: str = "memory://localhost"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()