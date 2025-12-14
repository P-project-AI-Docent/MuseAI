# backend/session_state.py

class SessionState:
    def __init__(self):
        # 예시 구조:
        # {
        #   "test": {
        #       "fallback_count": 1,
        #       "intent": "related_works",
        #       "last_question": "비슷한 작품 알려줘",
        #       "waiting_similar_choice": True
        #   }
        # }
        self.sessions = {}

    # -------------------------------------------------------
    # 여러 key=value 동시 업데이트
    # -------------------------------------------------------
    def update(self, session_id, **kwargs):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id].update(kwargs)

    # -------------------------------------------------------
    # 한 개의 key=value만 설정
    # -------------------------------------------------------
    def set(self, session_id, key, value):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id][key] = value

    # -------------------------------------------------------
    # key 값 가져오기
    # -------------------------------------------------------
    def get(self, session_id, key, default=None):
        return self.sessions.get(session_id, {}).get(key, default)

    # -------------------------------------------------------
    # key 숫자 증가 (fallback_count 등)
    # -------------------------------------------------------
    def increment(self, session_id, key):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        current = self.sessions[session_id].get(key, 0) + 1
        self.sessions[session_id][key] = current
        return current

    # -------------------------------------------------------
    # 특정 key만 초기화하거나 전체 세션 초기화
    # -------------------------------------------------------
    def reset(self, session_id, key=None):
        if session_id not in self.sessions:
            return

        # key만 초기화
        if key is not None:
            if key in self.sessions[session_id]:
                del self.sessions[session_id][key]
            return

        # 세션 전체 리셋
        self.sessions[session_id] = {}

    # -------------------------------------------------------
    # 모든 사용자 세션 초기화
    # -------------------------------------------------------
    def reset_all(self):
        self.sessions = {}


# 전역 인스턴스
session_state = SessionState()
