from flask_cors import CORS
from pytube import YouTube
from flask import Flask, render_template, Response, request, jsonify
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import os 
from datetime import datetime 
import csv
import pyrebase
app = Flask(__name__)
CORS(app) 
#connect to firebase
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

firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
db = firebase.database()
point = []
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
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
offset = 6
up=0
down=0


def youtube(url):
    print(url)
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension="mp4",res="720p").first()
    video_url = stream.url
    return  video_url

# Function to draw a line on the video frame
def draw_line(frame):
    global point, up, down, offset, person_up, counted_id
    if len(point) > 2:  # Limit the number of points stored
        point = point[-2:] 
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
            cv2.line(frame, (point[0][0],point[0][1]),(point[1][0],point[0][1]), 2)
    cvzone.putTextRect(frame,f'Out{up}',(50,60),2,2)
    cvzone.putTextRect(frame,f'In{down}',(50,160),2,2)
    return frame




def generate_frames(frame):
    global up, down
    video_capture = cv2.VideoCapture(camera_type)  # Change the index if your camera is not the default one
    frame_count = 0
    wait = 0

    while True:
        timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        file_name = f"{gait_name} {frame_count}.jpg"
        file_path = os.path.join(save_directory, file_name)

        success, frame = video_capture.read()

        if not success:
            break
        else:
            draw_line(frame)  # Draw the line and points on the frame

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            if wait % 600000 == 0:
                saved = cv2.imwrite(file_path, frame)
                if saved:
                    print(f"Frame {frame_count} saved as {file_name}")
                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([file_name, timestamp, up, down])
                    
                    # Upload frame image to Firebase Storage
                    storage.child(file_name).put(file_path)
                    
                    # Write frame number and timestamp to the Realtime Database
                    db.child("timestamps").child(f"counter{timestamp}").push({
                        "counter": file_name,
                        "Timestamp": timestamp, "countIN": up, "countOUT": down
                    })
                    
                    print(f"Frame {frame_count} data added to CSV and Firebase")
                
            frame_count += 1
            wait += 1000

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()
    
@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/', methods=['POST'])
def receive_data():
    global gait_name, camera_type,save_directory,csv_file_path
    data = request.json
    gait_name = data['input']
    camera_type = data['dropdown']
    try:
        camera_type = int(camera_type)
        print("my camera", camera_type)
    except:
        camera_type = youtube(camera_type)
        print("the url", camera_type)

    # Process the received data as needed
    print('Received data from client:', data['input'],data['dropdown'])
    
    # Path to the directory where folders will be created
    base_directory = gait_name
    save_directory = os.path.join(base_directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    csv_file_path = f'{gait_name}.csv'  # Replace with the path to your CSV file 
    # Check if the CSV file exists, if not, create it and write the header
    if not os.path.exists(csv_file_path):
        try: 
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([ 'File_Name' , 'Timestamp', 'countpeopleIn','countpeopleOut'])
            print(f"CSV file created at {csv_file_path}")
        except Exception as e:
            print(f"Error creating CSV file: {e}")  
   # Perform any additional processing or return a response if needed
    return jsonify({'status': 'success'})         
@app.route('/second_page')
def second_page():
    return render_template('second_page.html')

@app.route('/video_feed')
def video_feed():
    print(camera_type)
    return Response(generate_frames(camera_type), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_point', methods=['POST'])
def add_point():
    global point
    x = int(float(request.form['x']))
    y = int(float(request.form['y']))
    point.append((x, y))
    return 'Point added successfully'

if __name__ == '__main__':
    app.run(debug=True)