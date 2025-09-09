from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import os
import dotenv
from OpenAIChatbot import OpenAIChatbot
from EntityExtractor import EntityExtractor

dotenv.load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# dataclass : 클래스 정의 시 자동으로 클래스 속성을 생성
@dataclass
class DialogState:
    session_id: str
    entities: Dict[str, Any] = field(default_factory=dict)
    context_stack: List[Dict] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    last_updated:datetime = field(default_factory=datetime.now)

    def add_turn(self, user_input, bot_response, entities:Dict=None):
        turn = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'entities': entities or {}
        }
        self.conversation_history.append(turn)
        if entities:
            self.entities.update(entities)
        self.last_updated = datetime.now()

    def get_context(self, turns:int=3) -> List[Dict]:
        return self.conversation_history[-turns:] if self.conversation_history else []
    
    def clear_context(self):
        self.conversation_history = []
        self.entities = {}
        self.context_stack = []

# 대화 관리자 클래스
class ConversationManager:
    def __init__(self):
        self.sessions: Dict[str, DialogState] = {}
        self.entity_extractor = EntityExtractor()

    def create_session(self, user_id):
        session_id = user_id or str(uuid.uuid4())
        self.sessions[session_id] = DialogState(session_id=session_id)
        return session_id
    
    def get_session(self, session_id) -> Optional[DialogState]:
        return self.sessions.get(session_id)

    def process_message(self, session_id, user_input, chatbot):
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        dialog_state = self.sessions[session_id]

        entities = self.entity_extractor.extract_entities(user_input)
        entity_dict = {ent['label']: ent['text'] for ent in entities}
        # bot response
        context_prompt = self._build_context_prompt(dialog_state, user_input)
        response = chatbot.get_response(context_prompt)

        dialog_state.add_turn(
            user_input=user_input,
            bot_response=response,
            entities=entity_dict
        )

    def _build_context_prompt(self, dialog_state:Dict, current_input):
        context_info = []
        recent_context = dialog_state.get_context(turns=3)
        if recent_context:
            context_info.append('이전 대화 :')
            for turn in recent_context:
                context_info.append(f" 사용자 : {turn['user_input']}")
                context_info.append(f" 봇 : {turn['bot_response']}")

        if dialog_state.entities:
            context_info.append(f"기억된 정보 : {dialog_state.entities}")

        if dialog_state.user_profile:
            context_info.append(f"사용자 정보 : {dialog_state.user_profile}")

        context_info.append(f"현재 질문: {current_input}")

        return '\n'.join(context_info)

    def update_user_profiile(self, session_id, profile_data:Dict):
        if session_id in self.sessions:
            self.sessions[session_id].user_profile.update(profile_data)
    
    def get_conversation_summary(self, session_id):
        dialog_state = self.sessions.get(session_id)
        if not dialog_state or not dialog_state.conversation_history:
            return "대화 내역이 없습니다."
        
        summary_parts = []
        summary_parts.append(f"세션 ID: {session_id}")
        summary_parts.append(f"대화 턴 수: {len(dialog_state.conversation_history)}")
        summary_parts.append(f"추출된 개체: {dialog_state.entities}")

        return '\n'.join(summary_parts)

# 챗봇 구현
conv_manager = ConversationManager()

chatbot = OpenAIChatbot(api_key)
chatbot.set_system_prompt("당신은 친절한 비서입니다.")

session_id = conv_manager.create_session("user1233")

response1 = conv_manager.process_message(session_id, "안녕하세요.",chatbot)
print(f"봇1 : {response1}")
response2 = conv_manager.process_message(session_id, "제 이름은 김철수입니다.",chatbot)
print(f"봇2 : {response2}")

conv_manager.update_user_profiile(session_id, {"name": "김철수", "age": 30})

response3 = conv_manager.process_message(session_id, "아까 제가 뭐라고 했죠?",chatbot)
print(f"봇3 : {response3}")

response4 = conv_manager.process_message(session_id, "제 나이에 맞는 취미를 추천해주세요",chatbot)
print(f"봇4 : {response4}")

summary = conv_manager.get_conversation_summary(session_id)
print(f"대화 요약 :\n{summary}")