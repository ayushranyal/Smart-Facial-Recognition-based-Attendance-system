import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

# ====== CREATE REQUIRED FOLDERS ======
folders = ["TrainingImage", "TrainingImageLabel", "StudentDetails", "Attendance"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ====== WINDOW ======
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("900x600")
window.configure(bg="lightblue")

# ====== LABELS ======
tk.Label(window, text="Enter ID").place(x=100, y=100)
txt_id = tk.Entry(window)
txt_id.place(x=250, y=100)

tk.Label(window, text="Enter Name").place(x=100, y=150)
txt_name = tk.Entry(window)
txt_name.place(x=250, y=150)

message = tk.Label(window, text="", bg="lightblue", fg="red")
message.place(x=100, y=200)

# ====== TAKE IMAGES ======
def TakeImages():
    Id = txt_id.get()
    name = txt_name.get()

    # 1. Basic validation
    if not Id.isdigit() or not name.isalpha():
        message.config(text="Enter valid ID (number) and Name (letters)", fg="red")
        return

    # 2. Duplicate ID Check
    csv_path = "StudentDetails/StudentDetails.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Check if the ID already exists in the 'Id' column
            if int(Id) in df['Id'].values:
                message.config(text=f"Error: ID {Id} already exists!", fg="red")
                return
        except Exception as e:
            # If file is empty or corrupted, we just continue
            pass

    # 3. If ID is unique, proceed with camera
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sampleNum = 0
    title_take = "Registering Student"

    while True:
        ret, img = cam.read()
        if not ret: break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        cv2.putText(img, "Keep face steady and look straight", (50, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            sampleNum += 1
            face_img = gray[y:y+h, x:x+w]
            file_path = os.path.join("TrainingImage", f"{name}.{Id}.{sampleNum}.jpg")
            cv2.imwrite(file_path, face_img)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow(title_take, img)

        key = cv2.waitKey(1)
        if key == ord('c') or key == ord('C') or sampleNum >= 60 or cv2.getWindowProperty(title_take, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

    # 4. Save to CSV (Adding headers only if file is new)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a+", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Id", "Name"])
        writer.writerow([Id, name])

    message.config(text=f"Success: Registered {name} (ID: {Id})", fg="green")

# ====== TRAIN MODEL ======
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    Ids = []

    for file in os.listdir("TrainingImage"):
        if not file.endswith(".jpg"):
            continue

        path = os.path.join("TrainingImage", file)
        img = Image.open(path).convert('L')
        img_np = np.array(img, 'uint8')

        try:
            Id = int(file.split(".")[1])
        except:
            continue

        faces.append(img_np)
        Ids.append(Id)

    if len(faces) == 0:
        message.config(text="No images found to train")
        return

    recognizer.train(faces, np.array(Ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")

    message.config(text="Training completed!")

# ====== TRACK / ATTENDANCE ======
def TrackImages():
    if not os.path.exists("TrainingImageLabel/Trainner.yml"):
        message.config(text="Please train images first")
        return

    # Initialize recognizer and classifier
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Safe data loading: Handles headers and converts Id to Integer
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        df["Id"] = pd.to_numeric(df["Id"], errors='coerce')
        df = df.dropna(subset=["Id"])
        df["Id"] = df["Id"].astype(int)
    except Exception as e:
        message.config(text="Error reading StudentDetails.csv")
        return

    cam = cv2.VideoCapture(0)
    attendance = []

    while True:
        ret, img = cam.read()
        if not ret: break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 70:
                name_list = df.loc[df["Id"] == Id]["Name"].values
                name = name_list[0] if len(name_list) > 0 else "Unknown"

                # Get current time for the CSV row
                ts = time.time()
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                # Store data in order: Id, Time, Name
                attendance.append([Id, timeStamp, name])
                label = f"{Id}-{name}"
            else:
                label = "Unknown"

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Tracking (Press 'C' or 'c' to close this window)", img)

        # Exit logic: Press 'C' or 'c', or click the 'X' on the top rigth of window
        key = cv2.waitKey(1)
        if key == ord('C') or key == ord('c') or cv2.getWindowProperty("Tracking (Press 'C' or 'c' to close this window)", cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

    if attendance:
        # Create DataFrame
        df_att = pd.DataFrame(attendance, columns=["Id", "Time", "Name"])
        df_att = df_att.drop_duplicates(subset=['Id'], keep='first')
        
        # --- Filename Logic ---
        # Format: DD-MM-YYYY_Timing__Attendance.csv
        today_date = datetime.datetime.now().strftime('%d-%m-%Y')
        current_time = datetime.datetime.now().strftime('%I-%M%p') # %I is 12-hour clock, %p is AM/PM
        
        file_name = f"{today_date}_{current_time}__Attendance.csv"
        file_path = os.path.join("Attendance", file_name)
        
        # Save unique file for this session
        df_att.to_csv(file_path, index=False)

        message.config(text=f"Saved: {file_name}")
    else:
        message.config(text="No attendance recorded")

# ====== BUTTONS ======
tk.Button(window, text="Take Images", command=TakeImages).place(x=100, y=300)
tk.Button(window, text="Train Images", command=TrainImages).place(x=250, y=300)
tk.Button(window, text="Track Images", command=TrackImages).place(x=400, y=300)
tk.Button(window, text="End", command=window.destroy).place(x=550, y=300)

window.mainloop()