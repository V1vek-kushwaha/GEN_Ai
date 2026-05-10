from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel , Field
from typing import Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph , START ,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

class ChatState(BaseModel):
    messages: Annotated[list,add_messages]

llm = ChatGroq(model="openai/gpt-oss-20b")

def chatBotNode(state:ChatState) -> ChatState:
    res = llm.invoke(state.messages)
    # state.messages = [res]
    state.messages.append(res)  #for ui 
    return state

memory = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chatBot",chatBotNode)

graph.add_edge(START,"chatBot")
graph.add_edge("chatBot",END)

graph = graph.compile(checkpointer=memory)


# while True:
#     quary = input("User: ")
#     if quary.lower() in ["quit","bye","exit"]:
#         print("thanks for using Me !")

#     res = graph.invoke(
#         {"messages":[{"role":"user","content":quary}]},
#         config
#     )

#     ans = res["messages"][-1].content

#     print("Ai: ",ans)


## Web Ui 

st.header("Qna Bot With LangGraph")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message("role").markdown(message["content"])



prompt = st.chat_input("Ask Anythig ?")


if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").markdown(prompt)
    with st.chat_message("ai"):
        with st.spinner("Processing..."):
            responce = graph.invoke(
                {"messages": st.session_state.messages},
                {"configurable":{"thread_id":"my-bot-1"}}
                                    )
        
            result = responce["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role":"ai","content":result})