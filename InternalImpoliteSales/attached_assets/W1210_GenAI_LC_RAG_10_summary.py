# Filename: W1210_GenAI_LC_RAG_10_summary.py
#!pip install pypdf chromadb rapidocr-onnxruntime
#!pip install -U langchain-chroma

# ====================================
# 🚗 LangChain RAG Pipeline for Legal QA
# Refined Modular Version with Output Saving + CLI Mode + Model Selection
# ====================================

import os
import datetime, time
from dotenv import load_dotenv
from rich import print as pprint
from tqdm import tqdm

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

from langchain.chains.summarize import load_summarize_chain

# === Initialization Functions ===
def init_environment():
    load_dotenv(override=True)
    return os.getenv("OPENAI_API_KEY"), os.getenv("GEMINI_API_KEY")

def init_llms(openai_key, gemini_key, use_gemini=True):
    if use_gemini:
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    else:
        return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def load_and_split_pdf(pdf_path, chunk_size=500, chunk_overlap=20):
    docs = PyPDFLoader(file_path=pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return docs, splitter.split_documents(docs)

def save_output(question, context, answer, output_path="./output"):
    ensure_directory(output_path)
    filename = os.path.join(output_path, f"rag_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"[Query]\n{question}\n\n[Retrieved Context]\n{context}\n\n[Answer]\n{answer}\n")
    return filename


#使用 WebBaseLoader 讀取網頁內容。拆成長度為 200 字、重疊 10 字的段落。
def summarize_docs(llm):
    docs = WebBaseLoader("https://blog.langchain.dev/nvidia-nim/").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10).split_documents(docs)
    # Map - Reduce 架構摘要提示語（中文優化）。
    map_prompt = PromptTemplate.from_template("使用繁體中文和台灣用詞, 以下是一組文件串列：\n{text}\n請總結此文件")
    reduce_prompt = PromptTemplate.from_template("使用繁體中文和台灣用詞, 以下是文件內容：\n{text}\n請總結此內容")

    # 建立 map_reduce 摘要鏈，呼叫模型生成結果。
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=reduce_prompt)
    response = chain.invoke(chunks)['output_text']
    pprint(response)
    save_output(question="summarize the page content", context=response,answer=response, output_path="./output")

def print_assignment_info(student_id):
    print("\n-----------------------------")
    print(f"W12 Assignment 10 | Student ID: {student_id} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-----------------------------\n")

# === Main Entry Point ===
def main():

    student_id = "B11108067"  # ← 請在這裡填上你的學號

    for i in tqdm(range(300), desc="處理中"):
        time.sleep(1)  # 模擬等待

    openai_key, gemini_key = init_environment()
    llm = init_llms(openai_key, gemini_key)

    summarize_docs(llm)
    print_assignment_info(student_id)

if __name__ == "__main__":
    main()
