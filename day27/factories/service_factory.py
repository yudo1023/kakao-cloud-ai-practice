from abc import ABC, abstractmethod
from repositories.user_repository import UserRepository

class UserService:
    def __init__(self, repositoory:UserRepository):
        self.repository = repositoory

class EmailService:
    def __init__(self, smtp_server:str):
        self.smtp_server = smtp_server

class ServiceFactory(ABC):
    @abstractmethod
    def create_user_service(self) -> UserService:
        pass

    @abstractmethod
    def create_email_service(self) -> EmailService:
        pass

# 상용
class ProductionServiceFactory(ServiceFactory):
    def create_user_service(self) -> UserService:
        repository = UserRepository(connection='memory')
        return UserService(repository)
    
# 테스트 환경요 서비스 팩토리
class TestServiceFacctory():
    def create_user_service(self) -> UserService:
        repository = UserRepository(connection='test')
        return UserService(repository)

def get_service_factory() -> UserService:
    if 상용이면:
        ProductionServiceFactory()
    elif 테스트면:
        TestServiceFacctory()
