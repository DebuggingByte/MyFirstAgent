import streamlit as st
from langchain_core.messages import SystemMessage, BaseMessage
from create_llm_message import create_llm_msg

class WritingAgent:
    def __init__(self, model):
        self.model = model
        self.system_prompt = "You are a helpful assistant that can answer questions about writing."
        self.sessionHistory = []

    def get_response(self, user_input: str):
        msg = create_llm_msg(self.system_prompt, self.sessionHistory)
        llm_response = self.model.invoke(msg)

        return llm_response

    def writing_agent(self, user_input: str):
        full_response = self.get_response(user_input)

        return {
            "lnode": "writing_agent",
            "responseToUser": full_response,
            "category": "writing"
        }