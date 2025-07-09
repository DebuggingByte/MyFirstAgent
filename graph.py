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
        workflow.add_node("math", self.math_agent)
        workflow.add_node("reading", self.reading_agent)
        workflow.add_node("writing", self.writing_agent)
        workflow.add_node("general", self.general_agent)

        workflow.add_edge(START, "start")
        workflow.add_conditional_edges(
            "start",
            self.route_to_agent,
            {
                "math": "math",
                "reading": "reading", 
                "writing": "writing",
                "general": "general"
            }
        )
        workflow.add_edge("math", END)
        workflow.add_edge("reading", END)
        workflow.add_edge("writing", END)
        workflow.add_edge("general", END)

        self.workflow = workflow.compile()

    #Defining the route_to_agent function
    def route_to_agent(self, state: State) -> str:
        """Route to the appropriate agent based on category."""
        category = state.get("category", "general")
        # Ensure category is one of the valid options
        if category not in ["math", "reading", "writing", "general"]:
            category = "general"  # Default to general for unknown categories
        return category

    #Defining the initial_classifier function
    def initial_classifier(self, state: State) -> State:
        """Classify the user input to determine which agent should handle it."""
        classifier_prompt = """
        You are a helpful assistant that can classify user input into one of the following categories:
        - math: for mathematical questions, calculations, equations, numbers, geometry, algebra, etc.
        - reading: for questions about reading comprehension, literature, books, stories, text analysis, etc.
        - writing: for questions about writing, grammar, composition, essays, creative writing, etc.
        - general: for any other questions that don't fit the above categories (science, history, sports, etc.)
        
        Analyze the user input: "{user_input}"
        
        Respond with ONLY the category name (math, reading, writing, or general). Do not include any other text.
        """
        #Creating the message with user input included
        formatted_prompt = classifier_prompt.format(user_input=state["user_input"])
        msg = create_llm_msg(formatted_prompt, state.get("sessionHistory", []))
        #Invoking the model
        llm_response = self.model.invoke(msg)
        #Getting the category
        category = str(llm_response.content).strip().lower()
        
        # Validate and clean the category
        if category not in ["math", "reading", "writing", "general"]:
            # If the classifier didn't return a valid category, default to general
            category = "general"
        
        #Returning the state with all original values preserved
        result_state = {
            **state,  # Preserve all original state values
            "category": category,
            "lnode": "initial_classifier"
        }
        return result_state  # type: ignore

    def math_agent(self, state: State) -> State:
        """Handle math-related queries."""
        return self.math_agent_class.math_agent(state["user_input"], state.get("sessionHistory", []))  # type: ignore

    def reading_agent(self, state: State) -> State:
        """Handle reading-related queries."""
        return self.reading_agent_class.reading_agent(state["user_input"], state.get("sessionHistory", []))  # type: ignore

    def writing_agent(self, state: State) -> State:
        """Handle writing-related queries."""
        return self.writing_agent_class.writing_agent(state["user_input"], state.get("sessionHistory", []))  # type: ignore

    def general_agent(self, state: State) -> State:
        """Handle general queries that don't fit other categories."""
        user_input = state["user_input"].lower()
        
        # Check for introduction/greeting keywords
        intro_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
                         "who are you", "what are you", "introduce yourself", "tell me about yourself",
                         "what can you do", "start", "begin"]
        
        is_intro = any(keyword in user_input for keyword in intro_keywords)
        
        if is_intro:
            response = """Hello! I'm your AI Teacher Assistant, and I'm here to help you with your studies! 

I specialize in three main subjects:
• **Math**: I can help with equations, calculations, geometry, algebra, and problem-solving
• **Reading**: I can assist with reading comprehension, literature analysis, and text interpretation
• **Writing**: I can guide you with essays, grammar, composition, and creative writing

Just ask me any question related to these subjects, and I'll help you learn step by step. What would you like to work on today?"""
        else:
            response = "Sorry, but I can't answer that question. I'm specifically designed to help with math, reading, and writing questions. Please ask me something related to these subjects!"
        
        return {
            "lnode": "general_agent",
            "responseToUser": response,
            "category": "general",
            "sessionHistory": state.get("sessionHistory", []),
            "user_input": state["user_input"]
        }
