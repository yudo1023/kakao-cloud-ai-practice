from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = 'mysql+pymysql://사용자명:비밀번호@호스트:포트/데이터베이스명'

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocomit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == '__main__':
    try:
        with engine.connect() as connection:
            result = connection.execute(text('SELECT VERSION() as mysql_version'))
            version = result.fetchone()
            print(f"mysql 버전: {version[0]}")
            db = next(get_db())
            print(f"연결된 세션: {db}")
            result = db.execute(text('SELECT DATABASE() as current_db'))
            current_db = result.fetchone()
            print(f"현재 데이터베이스: {current_db[0]}")
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패 : {e}")
    finally:
        if 'db' in locals():
            db.close()
            print('데이터베이스 연결 종료')
