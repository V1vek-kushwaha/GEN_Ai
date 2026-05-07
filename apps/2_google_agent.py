from dotenv import load_dotenv
load_dotenv()



from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver  


llm = ChatGroq(model="openai/gpt-oss-120b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=InMemorySaver(),
    system_prompt="you are agent that can search any qustion on google"
)


while True:
    quary = input("user: ")
    if quary.lower() == "quit":
        print("Good Bye 👋")
        break

    responce = agent.invoke({"messages":[{"role":"user","content":quary}]},
                             {"configurable":{"thread_id":"1"}},
                            )
    print("AI: ",responce["messages"][-1].content)


