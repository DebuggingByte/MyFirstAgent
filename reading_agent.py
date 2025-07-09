import streamlit as st
from langchain_core.messages import SystemMessage, BaseMessage
from create_llm_message import create_llm_msg

class ReadingAgent:
    def __init__(self, model):
        self.model = model
        self.system_prompt = "You are a helpful assistant that can answer questions about reading."
        self.sessionHistory = []

    def get_response(self, user_input: str):
        msg = create_llm_msg(self.system_prompt, self.sessionHistory)
        llm_response = self.model.invoke(msg)

        return llm_response

    def reading_agent(self, user_input: str, session_history=None):
        if session_history is None:
            session_history = []
        
        msg = create_llm_msg(self.system_prompt, session_history)
        llm_response = self.model.invoke(msg)

        return {
            "lnode": "reading_agent",
            "responseToUser": llm_response.content,
            "category": "reading",
            "sessionHistory": session_history,
            "user_input": user_input
        }