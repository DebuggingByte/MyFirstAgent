import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph import TeacherAgent, State

def main():
    st.title("🤖 Teacher Assistant")
    st.markdown("Ask me questions about math, reading, or writing!")

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to learn about?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Initialize the teacher agent
                agent = TeacherAgent(st.secrets["OPENAI_API_KEY"])
                
                # Convert session messages to LangChain format
                session_history = []
                for msg in st.session_state.messages[:-1]:  # Exclude current message
                    if msg["role"] == "user":
                        session_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        session_history.append(AIMessage(content=msg["content"]))

                # Create initial state
                initial_state: State = {
                    "user_input": prompt,
                    "sessionHistory": session_history,
                    "lnode": None,
                    "category": None,
                    "responseToUser": None
                }

                # Run the workflow
                final_state = agent.workflow.invoke(initial_state)
                
                # Get the response
                if final_state.get("responseToUser"):
                    response = final_state["responseToUser"]
                    message_placeholder.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    message_placeholder.markdown("I'm sorry, I couldn't generate a response. Please try again.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                message_placeholder.markdown("Sorry, something went wrong. Please try again.")

if __name__ == "__main__":
    main()







