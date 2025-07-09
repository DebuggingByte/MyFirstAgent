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
    #Making sure lnode is optional because it is not always needed
    lnode: Optional[str]
    #Making sure category is optional because it is not always needed
    category: Optional[str]
    #Making sure sessionHistory is a list of BaseMessage because it is a list of messages
    sessionHistory: List[BaseMessage]
    #Making sure user_input is a string
    user_input: str
    #Making sure responseToUser is optional because it is not always needed
    responseToUser: Optional[str]


class Category(BaseModel):
    #Making sure category is a string
    category: str


class TeacherAgent():
    #Adding init function to initialize the model
    def __init__(self, api_key):
        #setting the model to the model specified in the secrets
        self.model = ChatOpenAI(model=st.secrets["model"], api_key=api_key)

        self.math_agent_class = MathAgent(self.model)
        self.reading_agent_class = ReadingAgent(self.model)
        self.writing_agent_class = WritingAgent(self.model)

        #Creating the workflow
        workflow = StateGraph(State)

        workflow.add_node("start", self.initial_classifier)
        workflow.add_node("math", lambda state: self.math_agent_class.math_agent(state["user_input"], state.get("sessionHistory", [])))
        workflow.add_node("reading", lambda state: self.reading_agent_class.reading_agent(state["user_input"], state.get("sessionHistory", [])))
        workflow.add_node("writing", lambda state: self.writing_agent_class.writing_agent(state["user_input"], state.get("sessionHistory", [])))

        workflow.add_edge(START, "start")
        workflow.add_conditional_edges(
            "start",
            self.route_to_agent,
            {
                "math": "math",
                "reading": "reading", 
                "writing": "writing"
            }
        )
        workflow.add_edge("math", END)
        workflow.add_edge("reading", END)
        workflow.add_edge("writing", END)

        self.workflow = workflow.compile()

    #Defining the route_to_agent function
    def route_to_agent(self, state: State) -> str:
        """Route to the appropriate agent based on category."""
        category = state.get("category", "teacher")
        return category if category else "teacher"

    #Defining the initial_classifier function
    def initial_classifier(self, state: State) -> State:
        """Classify the user input to determine which agent should handle it."""
        classifier_prompt = """
        You are a helpful assistant that can classify user input into one of the following categories:
        - math
        - reading
        - writing
        
        Respond with only the category name (math, reading, or writing).
        """
        #Creating the message
        msg = create_llm_msg(classifier_prompt, state.get("sessionHistory", []))
        #Invoking the model
        llm_response = self.model.invoke(msg)
        #Getting the category
        category = str(llm_response.content).strip().lower()
        #Returning the state
        return {
            "category": category,
            "lnode": "initial_classifier",
            "sessionHistory": state.get("sessionHistory", []),
            "user_input": state["user_input"],
            "responseToUser": state.get("responseToUser")
        }
