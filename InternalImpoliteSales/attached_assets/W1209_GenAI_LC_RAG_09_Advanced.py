# Filename: W1209_GenAI_LC_RAG_09_Advanced.py
# ====================================
# 🚗 LangChain RAG Pipeline for Legal QA
# Refined Modular Version with Output Saving + CLI Mode + Model Selection
# ====================================

# !pip install langchain_chroma

import os
import datetime, time
import argparse
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import (
    CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
)
from langchain_chroma import Chroma
from langchain_core.prompts import (
    ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import HumanMessage

# === Initialization Functions ===
def init_environment():
    load_dotenv(override=True)
    os.environ["USER_AGENT"] = "LangChain-RAG-Agent/1.0"
    return os.getenv("OPENAI_API_KEY"), os.getenv("GEMINI_API_KEY")

def init_llms(openai_key, gemini_key, use_gemini=False):
    if use_gemini:
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    else:
        return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

# 使用 PyPDFLoader 載入車輛法規 PDF。
# 用 RecursiveCharacterTextSplitter 拆成合理的語段（chunk）。
def load_and_split_pdf(pdf_path, chunk_size=500, chunk_overlap=20):
    docs = PyPDFLoader(file_path=pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return docs, splitter.split_documents(docs)

# 將拆段內容轉成向量（用 text-embedding-3-large 模型）。
# 存到本地向量庫 ./chroma_langchain_db 中。
def create_chroma_db(chunks, embedding_model, db_path="./chroma_langchain_db"):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path,
        collection_metadata={"hnsw:space": "cosine"}
    )

# 傳入問題 query，retriever 擷取最相關的段落。
# 用 prompt 包裝成 LLM 輸入。
def retrieve_and_ask(llm, retriever, query):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs if hasattr(doc, "page_content")])
    prompt = f"請根據以下內容回答問題:\n\n{context}\n\n問題: {query}"
    return llm.invoke([HumanMessage(content=prompt)]), context

def save_output(question, context, answer, output_path="./output"):
    ensure_directory(output_path)
    filename = os.path.join(output_path, f"rag_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"[Query]\n{question}\n\n[Retrieved Context]\n{context}\n\n[Answer]\n{answer}\n")
    return filename

def build_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template("請根據以下內容加上自身判斷回答問題:\n{context}\n問題: {question}")
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# 組合：Retriever → Prompt → LLM → OutputParser。
# 可支援 stream 回傳結果（逐字顯示）。
def wrap_chat_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template("請根據以下內容加上自身判斷回答問題:\n{context}\n問題: {question}")
    chain = prompt | llm | StrOutputParser()
    return RunnableParallel({"context": retriever, "question": RunnablePassthrough()}).assign(answer=chain)

def chat_interface(rag_chain, question, output_path="./output"):
    output_chunks = []
    print("\n[Response]")
    for chunk in rag_chain.stream(question):
        if isinstance(chunk, dict) and "answer" in chunk:
            value = chunk["answer"]
            if isinstance(value, list):
                output_chunks.extend(str(v) for v in value)
                print("".join(str(v) for v in value), end="", flush=True)
            else:
                output_chunks.append(str(value))
                print(value, end="", flush=True)
        else:
            for val in chunk.values():
                if isinstance(val, list):
                    output_chunks.extend(str(v) for v in val)
                    print("".join(str(v) for v in val), end="", flush=True)
                else:
                    output_chunks.append(str(val))
                    print(val, end="", flush=True)
    print("\n")
    combined = "".join(output_chunks)
    save_output(question=question, context="(streamed response, no context)", answer=combined, output_path=output_path)
    return combined

def print_assignment_info(student_id):
    print("\n-----------------------------")
    print(f"W12 Assignment 9 | Student ID: {student_id} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-----------------------------\n")

# === Main Entry Point ===
def main():

    student_id = "B11108067"  # ← 請在這裡填上你的學號

    parser = argparse.ArgumentParser(description="LangChain RAG CLI")
    parser.add_argument("--ask", type=str, help="Question to ask the legal QA system")
    parser.add_argument("--model", type=str, choices=["openai", "gemini"], default="gemini", help="Choose model: openai or gemini")
    args = parser.parse_args()

    openai_key, gemini_key = init_environment()
    use_gemini = args.model == "gemini"
    llm = init_llms(openai_key, gemini_key, use_gemini)

    docs, chunks = load_and_split_pdf("references/CarRegulations_True_or_False_Questions.pdf")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    db_path = "./chroma_langchain_db"
    ensure_directory(db_path)
    db = create_chroma_db(chunks, embeddings, db_path)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})


    if args.ask:
        chat_interface(wrap_chat_chain(llm, retriever), args.ask)
    else:
        print("請使用 --ask 參數輸入您的問題，例如: python script.py --ask '紅燈右轉會被罰款嗎？'")


    print_assignment_info(student_id)


if __name__ == "__main__":
    main()
