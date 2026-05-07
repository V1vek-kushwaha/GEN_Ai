from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-120b",streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]


if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []



agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer= st.session_state.memory,
    system_prompt="you are a ai agent who can search on google",
)

#### Building Web Interface

st.subheader("💬 QuickAnswer - Answer as the speed Thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


quary = st.chat_input("Ask Anything ?")

if quary:
    st.chat_message("user").markdown(quary)
    st.session_state.history.append({"role":"user","content":quary})

    responce = agent.stream(
        {"messages":[{"role":"user","content":quary}]},
        {"configurable":{"thread_id":"1"}},
        stream_mode="messages"
    )

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in responce:
            message = message + chunk[0].content
            space.write(message)

        st.session_state.history.append({"role":"ai","content":message})

