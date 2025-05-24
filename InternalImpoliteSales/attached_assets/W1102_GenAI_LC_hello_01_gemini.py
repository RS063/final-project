# Filename: W1102_GenAI_LC_hello_01_gemini.py
# 用 Gemini Flash 模型打一個中文問候，然後用不同方式印出回應
# Test Google Gemini API
#!pip install langchain
#!pip install -U langchain-community
#!pip install rich
#!pip install langchain_google_genai


import os, datetime
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_google_genai import ChatGoogleGenerativeAI

# 建立 Gemini LLM 物件
llm = ChatGoogleGenerativeAI( model="gemini-2.0-flash", temperature=0.7)

response = llm.invoke("你好, 使用繁體中文")
print(response)

response.pretty_print()
#
from rich import print as pprint
pprint(response)

# Assignment setting: Please enter your Student ID.
print("\n-----------------------------")
student_id = "B11108043"  # ← 請在這裡填上你的學號
print(f"W11 Assignment 2-1 | Student ID: {student_id} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-----------------------------\n")
