from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


import numpy as np
import faiss
import os

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGroq(
    temperature=0.6,
    model="llama-3.1-70b-versatile",
    api_key=os.getenv('GROQ_API_KEY'),
    streaming=True,
    callbacks=[BaseCallbackHandler()]
)

memory = ConversationSummaryMemory(
    llm=chat_model, memory_key="chat_history", input_key="question", return_messages=True)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_create_embeddings(uploaded_file):
    file_path = f"/tmp/{uploaded_file.name}"

    # Save the uploaded file content to a temporary file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    document_texts = [doc.page_content for doc in docs]
    document_embeddings = embedding_model.encode(
        document_texts, convert_to_tensor=False)

    document_embeddings = np.array(document_embeddings)

    faiss_index = create_faiss(document_embeddings)

    return docs, document_embeddings, faiss_index


def create_faiss(document_embeddings):
    dimension = document_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(document_embeddings))

    faiss.write_index(faiss_index, "faiss_index.index")
    return faiss_index


def generate_response(query, docs, faiss_index):
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # docs, document_embeddings, faiss_index = load_and_create_embeddings()

    # faiss_index = create_faiss(document_embeddings)

    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    vector_store = FAISS(
        index=faiss_index,
        embedding_function=embedding_model.encode,
        docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)}),
        index_to_docstore_id=index_to_docstore_id
    )

    retriever = vector_store.as_retriever()

    # memory = ConversationSummaryMemory(
    #     llm=chat_model,
    #     memory_key="chat_history",
    #     return_messages=True,
    #     max_token_limit=1000,
    #     input_key="query",
    #     output_key="result",
    #     summary_prompt="Summarize the key points of the conversation so far.",
    # )

    PROMPT_TEMPLATE = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are an AI designed to assist with answering questions strictly based on the provided context."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            SystemMessagePromptTemplate.from_template("{context}"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE,
                           "memory": memory},
    )

    for response in qa_chain.stream({"query": query}):
        return response['result']


# query = "hoe much money we spend for diverse suppliers"
# response = generate_response(query)
# print(response)
