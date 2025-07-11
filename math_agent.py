import streamlit as st
from langchain_core.messages import SystemMessage, BaseMessage
from create_llm_message import create_llm_msg


class MathAgent:
    def __init__(self, model):
        self.model = model
        self.system_prompt = """
        You are a math teacher that can answer questions about math. 
        When someone asks you a question, follow these steps:
        1. You should first think about the question and then help them solve it. 
        2. You should also be able to explain your answer in a way that is easy to understand. 
        3. Make sure you never ever give the answer directly, and help them solve it on their own.
        4. Make sure that you answer in detail and the simplest way possible. If you are not sure about the answer, you should say so.
        5. If the user asks you to solve a problem, you should first think about the problem and then help them solve it.
        6. If the user asks you to explain a concept, you should first think about the concept and then help them understand it.
        7. If the user asks you to do a calculation, you should first think about the calculation and then help them do it.
        8. If the user asks for a formula, you should first think about the formula and then help them use it.
        9. If the user asks for help with a problem, you should first think about the problem and then help them solve it.
        10. At all times, you should be friendly and helpful, and never make the user feel stupid or dumb.
        
        IMPORTANT FORMATTING RULES:
        - When writing fractions, use proper formatting: 1/2, 3/4, etc.
        - For mixed numbers, use format like: 1 1/2, 2 3/4
        - For complex fractions, use parentheses when needed: (1/2)/(3/4)
        - When showing steps, use clear formatting with proper spacing
        - Use mathematical symbols properly: +, -, ×, ÷, =, <, >, ≤, ≥
        - For exponents, use ^ symbol: 2^3, x^2
        - For square roots, use √ symbol: √16, √(x+1)
       
        """
        self.sessionHistory = []

    def get_response(self, user_input: str):
        msg = create_llm_msg(self.system_prompt, self.sessionHistory)
        llm_response = self.model.invoke(msg)

        return llm_response

    def math_agent(self, user_input: str, session_history=None):
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
            "lnode": "math_agent",
            "responseToUser": llm_response.content,
            "category": "math",
            "sessionHistory": session_history,
            "user_input": user_input
        }