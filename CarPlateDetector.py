from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
from ultralytics import YOLO
import easyocr as ea
import numpy as np
import time

# Load the YOLO model with a pretrained model file
model = YOLO(r'C:\Users\usuario\Desktop\Vision_computacional\ProyectoVision\ProyectoVision\best.pt')
# Create an instance of easyocr Reader for optical character recognition
reader = ea.Reader(["en"], gpu=True)

start_time = time.time()

# Function to detect plates in a video frame
def plate_detection(frame):
    # Process the frame with the YOLO model, adjusting the image size and confidence level
    results = model.predict(frame, imgsz=640, conf=0.4)

    # Display detection results on the frame
    annotations = results[0].plot()

    # Iterate through the detection results
    for result in results[0].boxes.data.tolist():
        # Unpack the data from each detection
        x1, y1, x2, y2, score, class_id = result
        # Convert the frame to a numpy array
        frame = np.asarray(frame)
        # Extract the plate region from the frame
        plate = frame[int(y1):int(y2), int(x1):int(x2)]
        # Use easyocr to read text in the plate region
        results_text = reader.readtext(plate, text_threshold=0.7, link_threshold=0.5)
        for res in results_text:
            # Extract the text from the plate
            plate_text = res[1]
            # Clear and update the text widget with the plate text
            text_entry.delete("1.0", "end")
            text_entry.insert("1.0", plate_text)

    end_time = time.time()
    print(end_time - start_time)
    # Return the annotated frame
    return annotations

# Function to start video capture from the webcam
def input_video():
    global cap
    # Initialize video capture
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # Start the display process
    display()

# Function to capture a frame from the video, process it, and update the GUI
def display():
    global cap
    # Read a frame from the video
    ret, frame = cap.read()
    # If the capture is successful, process and display the frame
    if ret == True:
        # Resize the frame for processing
        frame = imutils.resize(frame, width=640)
        # Call the plate detection function
        frame = plate_detection(frame)
        # Convert the frame to a format suitable for Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to a PIL image format
        im = Image.fromarray(frame)
        # Convert the PIL image to ImageTk
        img = ImageTk.PhotoImage(image=im)
        # Update the video label with the new image
        video_label.configure(image=img)
        video_label.image = img
        # Schedule the next update
        video_label.after(10, display)
    else:
        # Clear the GUI if video capture fails
        video_label.image = ""
        video_info_label.configure(text="")
        radio_button.configure(state="active")
        selected.set(0)
        end_button.configure(state="disabled")
        cap.release()

# Function to release video capture and reset the GUI
def finalize_clean():
    video_label.image = ""
    video_info_label.configure(text="")
    radio_button.configure(state="active")
    selected.set(0)
    # Release the video capture object
    cap.release()

# Initialize the cap variable for video capture
cap = None

# GUI setup
root = Tk()
video_info_label = Label(root, text="INPUT VIDEO", font="bold")
video_info_label.grid(column=0, row=0, columnspan=2)
selected = IntVar()
radio_button = Radiobutton(root, text="Live Video", width=20, value=2, variable=selected, command=input_video)
radio_button.grid(column=0, row=1)
video_info_label = Label(root, text="", width=20)
video_info_label.grid(column=0, row=2)

video_label = Label(root)
video_label.grid(column=0, row=3, columnspan=2)

end_button = Label(root, text="Plate")
end_button.grid(column=0, row=4, columnspan=2, pady=10, padx=10)

text_entry = Text(root, width=15, height=3, font=("Consolas", 12))
text_entry.grid(row=4, column=1, padx=5, pady=5)

root.mainloop()

