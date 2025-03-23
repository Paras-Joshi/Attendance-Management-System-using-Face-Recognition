import cv2
import numpy as np
import os
import csv
from datetime import datetime
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox

# Initialize face recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Paths for saving images and attendance
training_images_path = "TrainingImage"
students_data_file = "StudentsData.csv"
attendance_file = "Attendance.csv"

# Function to take images and store student data
def take_img(enrollment, name): 
    cam = cv2.VideoCapture(0)
    sample_num = 0
    os.makedirs(training_images_path, exist_ok=True)

    # Save enrollment and name in StudentsData.csv
    if not check_id_in_students_data(enrollment):
        with open(students_data_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([enrollment, name])

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        #Scale Factor : detect faces at different sizes in the image.
        for (x, y, w, h) in faces:
            sample_num += 1
            image_filename = os.path.join(training_images_path, f"{name}.{enrollment}.{sample_num}.jpg")
            #Extracts and saves the detected face from the grayscale image.
            cv2.imwrite(image_filename, gray[y:y+h, x:x+w])

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.imshow('Capturing Images - Press q to stop', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()

# Check if ID exists in StudentsData.csv
def check_id_in_students_data(id):
    if os.path.exists(students_data_file):
        with open(students_data_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == str(id):
                    return True
    return False

# Function to train images
def train_img():
    faces, ids = get_images_and_labels(training_images_path)
    recognizer.train(faces, np.array(ids))
    recognizer.save("trained_model.yml")

# Extract images and labels
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        try: # Handle potential image opening errors
            img = Image.open(image_path).convert('L')
            img_np = np.array(img, 'uint8')
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_np)
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w]) #Crops face from Image
                ids.append(id)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    return face_samples, ids

# Function for recognizing faces and marking attendance
def mark_attendance():
    try:
        recognizer.read("trained_model.yml")
    except Exception as e:
        print(f"Error loading trained model: {e}. Make sure you have trained the model first.")
        return

    cam = cv2.VideoCapture(0)
    recognized_ids = set() #Create a Set to Store Recognized IDs, Avoids duplicates
    confidence_threshold = 60 #minimum confidence score for a face to be considered "recognized."

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Could not capture frame.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            name = get_name_from_students_data(id) # Get the name here

            if confidence < confidence_threshold:
                if id not in recognized_ids:
                    if not check_id_in_attendance(id):
                        mark_attendance_in_csv(id, name)  # Use the retrieved name
                        recognized_ids.add(id)
                        cv2.putText(img, f"{name} - Present", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.putText(img, f"{name} - Already Marked", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(img, f"{name} - Present", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:  # Unknown face or low confidence
                if check_id_in_students_data(id): # Check if the id is present in the student data
                    retrieved_name = get_name_from_students_data(id)
                    if retrieved_name != "Unknown": # If the name is not unknown, then it is a registered student
                         cv2.putText(img, f"{retrieved_name} (Not Recognized)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                         cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                     cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.imshow('Attendance Marking - Press q to stop', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# Check if ID already marked in attendance
def check_id_in_attendance(id):
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == str(id):
                    return True
    return False

# Retrieve name from StudentsData.csv
def get_name_from_students_data(id):
    if os.path.exists(students_data_file):
        with open(students_data_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == str(id):
                    return row[1]
    return "Unknown"

# Mark attendance in a CSV file
def mark_attendance_in_csv(id, name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    with open(attendance_file, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([id, name, dt_string])



# Simple GUI interface
def take_images_gui():
    enrollment = enrollment_entry.get()
    name = name_entry.get()

    if not enrollment or not name:
        messagebox.showerror("Error", "Enrollment and Name are required!")
        return

    try:
        int(enrollment)  # Check if enrollment is a valid integer
    except ValueError:
        messagebox.showerror("Error", "Invalid Enrollment Number!")
        return

    take_img(enrollment, name)
    messagebox.showinfo("Success", "Images captured successfully!")
    enrollment_entry.delete(0, tk.END)
    name_entry.delete(0, tk.END)



def train_model_gui():
    train_img()
    messagebox.showinfo("Success", "Model trained successfully!")

def mark_attendance_gui():
    mark_attendance()


# Tkinter GUI setup
window = tk.Tk()
window.title("Attendance Management System")
window.geometry("700x500")
window.resizable(True, True)

# Style configuration (modern look)
style = ttk.Style()
style.theme_use('clam')

# Color palette
BACKGROUND_COLOR = "#e8f0f7"  # Soft blue
FRAME_COLOR = "#ffffff"        # Pure white
BUTTON_COLOR = "#1e88e5"       # Light blue
BUTTON_HOVER_COLOR = "#1565c0" # Darker blue for hover
BUTTON_TEXT_COLOR = "#ffffff"   # White
LABEL_COLOR = "#333333"        # Dark gray
ENTRY_COLOR = "#f5f5f5"        # Light gray
TEXT_COLOR = "#212121"         # Almost black

# Apply colors and fonts to the style
window.configure(bg=BACKGROUND_COLOR)
style.configure("TLabelframe", background=FRAME_COLOR, foreground=LABEL_COLOR, font=('Helvetica', 14, 'bold'))
style.configure("TFrame", background=FRAME_COLOR)
style.configure("TButton", background=BUTTON_COLOR, foreground=BUTTON_TEXT_COLOR, font=('Helvetica', 12, 'bold'), padding=10)
style.configure("TLabel", background=FRAME_COLOR, foreground=LABEL_COLOR, font=('Helvetica', 12))
style.configure("TEntry", background=ENTRY_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 12))

# Take Images Frame
frame_padding = 30
take_images_frame = ttk.LabelFrame(window, text="Take Images", padding=frame_padding)
take_images_frame.pack(pady=30, padx=30, fill="both", expand=True)

# Enrollment
enrollment_label = ttk.Label(take_images_frame, text="Enrollment:")
enrollment_label.grid(row=0, column=0, padx=10, pady=15, sticky="w")
enrollment_entry = ttk.Entry(take_images_frame)
enrollment_entry.grid(row=0, column=1, padx=10, pady=15, sticky="ew")

# Name
name_label = ttk.Label(take_images_frame, text="Name:")
name_label.grid(row=1, column=0, padx=10, pady=15, sticky="w")
name_entry = ttk.Entry(take_images_frame)
name_entry.grid(row=1, column=1, padx=10, pady=15, sticky="ew")

# Take Images Button
take_button = ttk.Button(take_images_frame, text="Take Images", command=take_images_gui)
take_button.grid(row=2, column=0, columnspan=2, pady=20)

take_images_frame.columnconfigure(1, weight=1)

# Buttons for Train Model and Mark Attendance
train_button = ttk.Button(window, text="Train Model", command=train_model_gui)
train_button.pack(pady=10, padx=30, fill="x")

mark_button = ttk.Button(window, text="Mark Attendance", command=mark_attendance_gui)
mark_button.pack(pady=10, padx=30, fill="x")

# Main loop
window.mainloop()