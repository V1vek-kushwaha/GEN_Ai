from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


### data in st session

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None 

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):

    ## load the document
    loader = PyPDFDirectoryLoader(path)
    doc = loader.load()

    ## split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    docs = splitter.split_documents(documents=doc)

    ## embeddings and vector DB
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    vector_store = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    ## create a agent - tool ,llm , prompt
    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrive_context(quary:str):
        """ retrive documets relevant to a quary from the knowledge base. """
        context = ""
        
        docs = vector_store.similarity_search(query=quary,k=3)
        for doc in docs:
            context += doc.page_content + "\n\n"
        
        
        return context

    system_prompts = """
        you are a helpful assistent that answers questions using retrive
        my knowledge base consists of the details form the uploaded documents,
        always use the `retrive_context` tool for questions requiring extrnal knowladge 
        If the answer is not related to the pdf document, say:"This information is not related of uploaded pdf.
        """

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrive_context],
        system_prompt=system_prompts,
        checkpointer=memory
    )


    st.session_state.agent = agent
    st.session_state.document_uploaded = True
    st.session_state.agent = agent


### upload ui 
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF files",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
            
            process_document(path)
            st.rerun()



## chat ui

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    quary = st.chat_input("Ask anything related to uploaded documents..")
    if quary:
        st.session_state.messages.append({"role":"user","content":quary})
        st.chat_message("user").markdown(quary)
        responce = st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":quary}]},
            {"configurable":{"thread_id":1}}
        )

        answer = responce["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})






# while True:
#     query = input("User:")
#     if query.lower() == "quit":
#         break

#     response = agent.invoke(
#         {"messages":[{"role":"user","content":query}]},
#         {"configurable":{"thread_id":1}}
#         )
#     result = response["messages"][-1].content

#     print("AI:" ,result)
