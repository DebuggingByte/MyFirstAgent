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
        
        # Create a message that includes the user's question
        from langchain_core.messages import HumanMessage
        
        # Create messages with system prompt, session history, and current user input
        messages = []
        messages.append(SystemMessage(content=self.system_prompt))
        messages.extend(session_history)
        messages.append(HumanMessage(content=user_input))
        
        llm_response = self.model.invoke(messages)

        return {
            "lnode": "reading_agent",
            "responseToUser": llm_response.content,
            "category": "reading",
            "sessionHistory": session_history,
            "user_input": user_input
        }