from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


st.title("🤖 AskBuddy -  AI QnA BOT")
st.markdown("my QnA bot with langchain and google Gen Ai")

if "messages" not in st.session_state:
    st.session_state.messages = []



for messages in st.session_state.messages:
    role = messages["role"]
    content = messages["content"]
    st.chat_message(role).markdown(content)


quary = st.chat_input("Ask anything ?")
if quary:
    st.session_state.messages.append({"role":"user", "content":quary})
    st.chat_message("user").markdown(quary)
    res = llm.invoke(quary)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai", "content":res.content})

# while True:
#     quary = input("user: ")

#     if quary.lower in ["quit","exit","bye"]:
#         print("Good Bye")
#         break

#     result = llm.invoke(quary)
#     print("AI: ",result.content, "\n")