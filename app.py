import base64
from flask_cors import CORS
from pytube import YouTube
from flask import Flask, render_template, Response, request, jsonify
import cv2
import pandas as pd
import streamlink
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import os 
from datetime import datetime 
import csv
import pyrebase
from crowdDensity import counter
import numpy as np

app = Flask(__name__)
CORS(app) 


# Add firebase configuration to connect firebase and upload data in it
firebaseConfig = {
  "apiKey": "AIzaSyDWvrfySM8YXYa6AWvmglGfQwgeccaf7WQ",
  "authDomain": "camera-c-97e7e.firebaseapp.com",
  "databaseURL": "https://camera-c-97e7e-default-rtdb.firebaseio.com",
  "projectId": "camera-c-97e7e",
  "storageBucket": "camera-c-97e7e.appspot.com",
  "messagingSenderId": "150541285688",
  "appId": "1:150541285688:web:56a9be67cf0d1fa80169cd",
  "measurementId": "G-KTFCLTGHT1"
}
# Intilaize firebase with the provided configuration (firebaseConfig)

firebase = pyrebase.initialize_app(firebaseConfig)
# Access firebase storage service

storage = firebase.storage()
# Access firebase realtime database service

db = firebase.database()

# Load Yolo Model (Pretrained Model)

model = YOLO('yolov8s.pt')
point = []
person_down = []
counted_id = []
person_up = []
in_line = []
i = 1
wait = 0
point = []
count = 0
person_down = []
tracker = Tracker()
counted_id = []
person_up = []

# Read Coco.txt to get class names objects for that we can detect it using yolo model

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
offset = 6
up=0
down=0
total = 0



# Function that return url for youtube videos 
def youtube(url):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension="mp4",res="720p").first()
    video_url = stream.url
    return  video_url



# Function that return url for live streaming (Facebook - Twitter - Youtube) 
def stream(url):
    streams = streamlink.streams(url)
    print("url: ", url)
    if streams:
        print("streams: ", streams) 
        best_stream = streams["best"]
        print("best_stream.url: ", best_stream.url) 
        return best_stream.url
    else:
        return None

# Function to draw a line on the video frame
def draw_line(frame):
    global point, up, down, offset, person_up, counted_id
    if len(point) > 2:  # Limit the number of points stored
        point = point[-2:] 
        print(point)
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    # cv2.line(frame, (50, 50), (200, 200), (0, 255, 0), 5)

    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])
       
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
    #box axis and id
        x3, y3, x4, y4, id = bbox
        
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        # drow circle in the center of the box
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
    

        # if the center point in the line
        if len(point) == 2: 
            if point[0][1] <(y3 + offset ) and point[0][1] >(y3-offset):
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255))
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                # check if it calulated or not
                if id not in in_line:
                    in_line.append(id)
            # if the point in the line        
            else:
                # if the point go across the line and not counted 
                if id not in counted_id and id in in_line:
                    # if the point above line 
                    if  y3>(point[0][1] + offset+6):
                        up= up+1
                        in_line.remove(id)
                        counted_id.append(id)
                    # if the point under line
                    elif y3 <(point[0][1]-offset-6):
                        down = down+1
                        in_line.remove(id)
                        counted_id.append(id)
            cv2.line(frame, (point[0]),(point[1]), 100,4)
            
            
    # Draw  2 text boxes for people passes line if there is in , out
    # Draw text box for total in&out people        
    cvzone.putTextRect(frame,f'Out{up}',(50,60),2,2)
    cvzone.putTextRect(frame,f'In{down}',(50,130),2,2)
    
    total = up+down
    cvzone.putTextRect(frame,f'Total{total}',(50,200),2,2)

    return frame




def generate_frames(frame):
    global up, down
    # Open video
    video_capture = cv2.VideoCapture(camera_type)  # Change the index if your camera is not the default one
    frame_count = 1
    wait = 0

    while True:
        timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        file_name = f"{gait_name}{frame_count}.jpg"
        
       
        # Return frame of video 
        success, frame = video_capture.read()

        if not success:
            break
        else:
            width = 700
            height = 500
            # Resize frame
            frame = cv2.resize(frame, (width, height))
            draw_line(frame)  # Calling funcation that draw the line and points on the frame
            _, encoded_frame = cv2.imencode('.jpg', frame)
            frame_bytes = encoded_frame.tobytes()
        

            # Wait for 10 min to upload data to firebase
            if wait %  600000== 0:         
                
                # Upload the frame to Firebase Storage
                storage_ref = storage.child(file_name)
                firebase_path = storage_ref.put(frame_bytes)
                saved = firebase_path
                
                if saved:
                    print(f"Frame {frame_count} saved as {file_name}")            
                    # Write frame number and timestamp to the Realtime Database
                    db.child("timestamps").child(f"counter{timestamp}").push({
                        "image_name": file_name,
                        "Timestamp": timestamp, "countIN": up, "countOUT": down, "count":total,"from":gait_name
                    })
                    
                    print(f"Frame {frame_count} data added to Firebase")
                
            frame_count += 1
            wait += 1000

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()
    
@app.route('/')
# Function that return home page for our website    
def index():
    return render_template('Home.html')

@app.route('/', methods=['POST'])
# Funcation for recieving data that have been submited in home page   

def receive_data():
    global gait_name, camera_type 
    # Get data that have submited by user (gate name and camera type (port [0--4] or URL) )           
    data = request.json
    print(data)
    # Get gate name
    gait_name = data['input']
    if data['dropdown'] == 'URL':
        # Get Url
        camera_type = data['url']
        try :
            # Calling function that return url from live streaming (Facebook - Twitter - Youtube)
            camera_type = stream(camera_type)
            print("the url", camera_type)
        except :
            # Calling function that return url from youtube videos or live   
            camera_type = youtube(camera_type)
            print("the url", camera_type)                
    else:
        camera_type = data['dropdown']
        # Get camera port number (0--4) in case using static camera
        camera_type = int(camera_type)
    # print(data)
    # # print(data['url'])
    # print(data['dropdown'])
    # print("----------------------------------------------------------")
    # print(data)
    # # print(data['url'])
    # print(data['dropdown'])
    # print("----------------------------------------------------------")


    # Process the received data as needed
    # print('Received data from client:', data['input'],data['dropdown'])
   # Perform any additional processing or return a response if needed
    return jsonify({'status': 'success'})         
       
@app.route('/second_page')
# Function to go from home page to counting with line page    
def second_page():
    # Add gate name counting with line page as header
    gate_name = gait_name
    return render_template('second_page.html', gateNameInput=gate_name)


@app.route('/third_page')   
# Function to go from home page to crowded counting page    
def third_page():
    # Add gate name crowded counting page as header
    gate_name = gait_name
    return render_template('third_page.html', gateNameInput=gate_name)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    try:
        global captured_image_data

        # Receive the image data from the client
        data = request.json
        captured_image_data = data.get('image_data')

        # Decode the base64-encoded image data
        image_bytes = base64.b64decode(captured_image_data.split(',')[1])

        # Convert the image data to a NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode the NumPy array to a CV2 image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform processing on the captured image data if needed
        img,county  = counter(frame,gait_name)
        timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        file_name = f"{gait_name}{timestamp}.jpg"
        storage_ref = storage.child(file_name)
        firebase_path = storage_ref.put(base64.b64decode(img))
                   
        # Write frame number and timestamp to the Realtime Database
        db.child("timestamps").child(f"counter{timestamp}").push({
            "image_name": file_name,
            "Timestamp": timestamp, "count":county, "from":gait_name
        })
                    
            

        # Send back the processed image data
        return jsonify({'image_data': img})

    except Exception as e:
        error_message = f"Error processing image: {e}"
        print(error_message)

        # Log the error to a file or your preferred logging mechanism
        with open('error_log.txt', 'a') as log_file:
            log_file.write(f"{datetime.now()}: {error_message}\n")

        return jsonify({'error': f"Internal server error: {e}"}), 500


@app.route('/video_feed')
# Function that returns a response object for video streaming using generate_frames function
def video_feed():
    print(camera_type)
    return Response(generate_frames(camera_type), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add_point', methods=['POST'])
# Function for get coordinates for draw the line 
def add_point():
    global point
    # Get coordinates for x & y points using righ click of cursor   
    x = int(float(request.form['x']))
    y = int(float(request.form['y']))
    point.append((x, y))
    return 'Point added successfully'




if __name__ == '__main__':
    app.run(debug=True)