# Filename: W1105_GenAI_LC_MemoryCache_04.py
# MemoryCache : 可以即問即答的小問答機
import os
import datetime
from dotenv import load_dotenv
from rich import print as pprint
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# 載入環境變數
load_dotenv(override=True)

# 設定 LangChain 快取機制 (使用記憶體快取)
set_llm_cache(InMemoryCache())


def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        streaming=True,
        cache=True
    )


def ask_question(llm, question):
    try:
        response = llm.invoke(question)
        pprint("\nLLM 回答內容：")
        pprint(response.content)
        return response.content
    except Exception as e:
        print(f"呼叫 LLM 發生錯誤: {e}")
        return None


def save_to_file(student_id, question, answer):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"W11_Assignment5_{student_id}_{datetime.datetime.now().strftime('%Y%m%d')}.txt")
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"W11 Assignment 5 | Student ID: {student_id} | {datetime.datetime.now().strftime('%Y%m%d')}\n")
            file.write("-----------------------------\n")
            file.write(f"問題：{question}\n\n")
            file.write(f"回答：{answer}\n")
        print(f"已成功儲存到檔案：{filename}")
    except Exception as e:
        print(f"儲存檔案時發生錯誤: {e}")


def print_assignment_info(student_id):
    print("\n-----------------------------")
    print(f"W11 Assignment 5 | Student ID: {student_id} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-----------------------------\n")


def main():
    llm = create_llm()
    user_question = input("請輸入你想問的問題（請用繁體中文）： ")
    answer = ask_question(llm, user_question)

    student_id = "B111008043"  # ← 請在這裡填上你的學號
    print_assignment_info(student_id)

    if answer:
        save_to_file(student_id, user_question, answer)

# 若本程式被執行，視為程式的進入點，則執行 main()
if __name__ == "__main__":
    main()


# 進階思考題: 如何支援連續輸入多個問題（每一題自動累加在同一個檔案，直到你輸入"結束"為止）！