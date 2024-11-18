from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory  

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome! Ask anything you want about mental health.").send()

    model = Ollama(model="mistral")
    memory = ConversationBufferMemory(return_messages=True)  

    cl.user_session.set("model", model)
    cl.user_session.set("memory", memory)  

@cl.on_message
async def on_message(message: cl.Message):
    model = cl.user_session.get("model")  
    memory = cl.user_session.get("memory") 

    history = memory.load_memory_variables({})['history']
    formatted_history = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a very knowledgeable doctor who provides short answers to mental health questions."),
            ("human", formatted_history),  
            ("human", "{question}"),  
        ]
    )

    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)
