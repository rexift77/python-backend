import os
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import random
from typing import List
from contants import SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE")
OPENAI_MODEL_NAME = os.environ.get("GPT_MODEL")

def get_response(messages: List[any]):
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL_NAME,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    pc = Pinecone(api_key=PINECONE_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index = pc.describe_index(PINECONE_INDEX)

    pinecone_namespace = PINECONE_NAMESPACE
    
    SYSTEM_TEMPLATE = "Answer the user's questions based on the below context.\n" + SYSTEM_PROMPT + """ 
        <context>
        {context}
        </context>
    """

    vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_KEY, index_name=PINECONE_INDEX, embedding=embeddings,
                                      namespace=pinecone_namespace)
    retriever = vectorstore.as_retriever()

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant "
                "to the conversation. Only respond with the query, nothing else.",
            ),
        ]
    )

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            # If only one message, then we just pass that message's content to retriever
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
        query_transform_prompt | chat | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )

    history = []
    for item in messages:
        if item['type'] == "user":
            history.append(HumanMessage(content=item['text']))
        if item['type'] == "bot":
            history.append(AIMessage(content=item['text']))

    stream = conversational_retrieval_chain.stream(
        {
            "messages": history,
        }
    )

    all_content = ""
    for chunk in stream:
        for key in chunk:
            if key == "answer":
                all_content += chunk[key]
                processed_chunk = chunk[key].replace("\n", "\\n").replace("\t", "\\t")
                yield f'data: {processed_chunk}\n\n'