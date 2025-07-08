
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from embeddings.embeddings import Embeddings

# Load
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split
def get_text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    return splitter.split_text(raw_text)

# Embed and Store
def get_vectorstore_and_BM25(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(raw_text)

    load_dotenv()  
    api_key = os.getenv("OPENAI_API_KEY")

    embeddings = Embeddings(model_name="openai", api_key=api_key).get_embedding_function()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    bm25 = BM25Retriever.from_texts(chunks) 
    return vectorstore, bm25, chunks


# Retrieve
def get_context_retriever_chain(vectorstore, bm25_retriever):
    llm = ChatOpenAI()
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    ensemble = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    return create_history_aware_retriever(llm, ensemble, prompt) #This line only creates a retriever chain object. It's like preparing a tool that can generate a query based on chat history and then pass it to the ensemble retriever.

# Prompt and LLM
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are an intelligent AI assistant helping the user answer questions about an uploaded document. "
        "Use the provided context to answer as accurately and helpfully as possible.\n\n"
        "If the answer is clearly present in the context, respond directly.\n"
        "If the answer requires understanding or summarizing multiple parts of the context, you are allowed to infer and generate a helpful answer.\n"
        "Only say: 'Sorry, I couldn't find information related to your question in the uploaded document.' "
        "if the context has no relevant content at all.\n\n"
        "Context:\n{context}"),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(retriever_chain, create_stuff_documents_chain(llm, prompt))


# Question and Answer
def get_response(user_query, chat_history, vectorstore, bm25_retriever):
    retriever_chain = get_context_retriever_chain(vectorstore, bm25_retriever) # Gets back the most relevant document chunks pipeline
    qa_chain = get_conversational_rag_chain(retriever_chain)
    result = qa_chain.invoke({
        "chat_history": chat_history,
        "input": user_query
    })
    return result["answer"], result.get("context", [])
