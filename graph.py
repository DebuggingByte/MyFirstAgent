import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, TypedDict, Annotated, Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from math_agent import MathAgent
from reading_agent import ReadingAgent
from writing_agent import WritingAgent
from create_llm_message import create_llm_msg



class State(TypedDict):
    lnode: Optional[str]
    category: Optional[str]
    sessionHistory: List[BaseMessage]
    user_input: str
    responseToUser: Optional[str]


class Category(BaseModel):
    category: str


class TeacherAgent():
    def __init__(self, api_key):
        self.model = ChatOpenAI(model=st.secrets["model"], api_key=api_key)

        self.math_agent_class = MathAgent(self.model)
        self.reading_agent_class = ReadingAgent(self.model)
        self.writing_agent_class = WritingAgent(self.model)

        workflow = StateGraph(State)

        workflow.add_node("start", self.initial_classifier)
        workflow.add_node("math", lambda state: self.math_agent_class.math_agent(state["user_input"]))
        workflow.add_node("reading", lambda state: self.reading_agent_class.reading_agent(state["user_input"]))
        workflow.add_node("writing", lambda state: self.writing_agent_class.writing_agent(state["user_input"]))

        workflow.add_edge(START, "start")
        workflow.add_conditional_edges(
            "start",
            self.route_to_agent,
            {
                "math": "math",
                "reading": "reading", 
                "writing": "writing",
            }
        )
        workflow.add_edge("math", END)
        workflow.add_edge("reading", END)
        workflow.add_edge("writing", END)

        self.workflow = workflow.compile()

    def route_to_agent(self, state: State) -> str:
        """Route to the appropriate agent based on category."""
        category = state.get("category", "teacher")
        return category if category else "teacher"

    def initial_classifier(self, state: State) -> State:
        """Classify the user input to determine which agent should handle it."""
        classifier_prompt = """
        You are a helpful assistant that can classify user input into one of the following categories:
        - math
        - reading
        - writing
        
        Respond with only the category name (math, reading, or writing).
        """
        msg = create_llm_msg(classifier_prompt, state.get("sessionHistory", []))
        llm_response = self.model.invoke(msg)
        category = str(llm_response.content).strip().lower()
        
        return {
            "category": category,
            "lnode": "initial_classifier",
            "sessionHistory": state.get("sessionHistory", []),
            "user_input": state["user_input"],
            "responseToUser": state.get("responseToUser")
        }
