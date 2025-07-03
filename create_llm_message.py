import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

def create_llm_message(system_prompt: str, sessionHistory: list[BaseMessage]) -> list[BaseMessage]:
    resp = [] 
    resp.append(SystemMessage(content="system_prompt"))
    resp.extend(sessionHistory)
    return resp
    
    