import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# 模型載入
model = tf.keras.models.load_model("model.h5")
class_names =  ["寶特瓶",
    "衛生紙",
    "玻璃",
    "紙類",
    "便當盒",
    "手搖飲料杯",
    "廢電器",
    "鋁罐",
    "藥"    ]

class_info = {
    "寶特瓶": "我是寶特瓶 ♻️，請丟到資源回收喔！記得壓扁再回收！",
    "鋁罐": "我是鋁罐～請我進回收桶 ♻️",
    "玻璃": "我是玻璃！請小心丟到玻璃回收箱喔～",
    "紙類": "我是乾淨紙類 🧻 請回收我～",
    "衛生紙": "我是用過的衛生紙，要丟到一般垃圾桶唷",
    "便當盒": "我是吃完的便當盒 🍱 請分類看看可不可以洗乾淨回收",
    "手搖飲料杯": "我是手搖杯 🧋 請分開杯蓋、封膜再回收！",
    "藥": "我是藥品 💊 請回收至藥物回收箱",
    "廢電器": "我是廢電器 🧯 請送去資源回收站處理！"
}



# 主題顏色與字體
bg_color = "#E0FFE0"
btn_color = "#A3D977"
font_main = ("Comic Sans MS", 16)
font_title = ("Comic Sans MS", 22, "bold")

# 建立主視窗
window = tk.Tk()
window.title("垃圾分類小幫手 ♻️")
window.configure(bg=bg_color)
window.geometry("700x700")

# 標題文字
title = tk.Label(window, text="我是你的垃圾分類小幫手 ♻️", font=font_title, bg=bg_color)
title.pack(pady=10)

# 攝影機畫面
camera_label = tk.Label(window)
camera_label.pack()

# 分類結果文字
result_label = tk.Label(window, text="", font=font_main, bg=bg_color)
result_label.pack(pady=10)

# 開啟攝影機
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    window.after(10, update_frame)

def capture_and_classify():
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("錯誤", "無法取得相機畫面")
        return
    image = cv2.resize(frame, (224, 224)) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    idx = np.argmax(prediction)
    label = class_names[idx]

    # 顯示分類說明
    if label in class_info:
        result_label.config(text=class_info[label])
    else:
        result_label.config(text=f"這是：{label}（分類說明待補）")

# 按鈕
btn_frame = tk.Frame(window, bg=bg_color)
btn_frame.pack(pady=10)

detect_btn = tk.Button(btn_frame, text="📷 開始辨識", font=font_main, bg=btn_color, command=capture_and_classify)
detect_btn.grid(row=0, column=0, padx=20)

quit_btn = tk.Button(btn_frame, text="❌ 離開", font=font_main, bg="#FFB6B6", command=window.quit)
quit_btn.grid(row=0, column=1, padx=20)

update_frame()
window.mainloop()
cap.release()
