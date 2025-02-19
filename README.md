This project is all about creating and testing a YOLOV8 model for detecting vehciles and traffic lights in CARLA simulator. 
1. Dataset Collection : Any dataset of your choice can be used. Datasets can be 
found from the below links: 
https://universe.roboflow.com/project-tzzs1/video1-walrf/dataset/13 
https://www.kaggle.com/datasets/amitkumargurjar/car-detection-and-tracking
dataset 
Some of the images were collected from simulator itself. 
2. Annotation, Augmentation and Labelling : Create a roboflow account, upload 
the dataset in a new project. The nest step is to annotate , label and apply 
augmentation techniques to the images. After all the steps the dataset is ready to 
be used for model training (Download the dataset inYOLOv8 format). 
3. Google Colab for model training :  
• !pip install roboflow  
• Use the download code for the dataset in Google Colab 
• %cd /content/the name of the folder in which the dataset is mounted 
• from ultralytics import YOLO 
# Create a YOLOv8 model for training 
model = YOLO("yolov8n.pt")  # yolov8n.pt is the smallest version; you can 
also use yolov8s.pt, yolov8m.pt, etc. 
• # Train on your downloaded dataset 
model.train(data="/content/dataset folder name/data.yaml", epochs=50, 
imgsz=640) 
• Now the model is ready after training and can be downloaded  
from google.colab import files 
f
 iles.download("//content/dataset /runs/detect/train2/weights/best.pt") 
• Use the codes for validation and testing as well 
4. Carla Installation :  
• CARLA 0.9.11 version is used in the setup  
• For creating the virtual environment Run Anaconda prompt as 
administrator 
• This command can be used for creating the environment :  conda create 
n myenv python=3.7. The version of python can be seen in the folder 
WindowsNoeditor/PythonAPI/carla/dist/ 
• Activate the environment and move to the location WindowsNoeditor/ and 
run the CARLA.exe by using Carla.exe command 
• Now the environment is ready 
5. Final integration 
• Download the code from the github repo: 
https://github.com/Josey0108/Vehicle-and-Traffic-Light-detection-in
CARLA.git 
• Navigate to the WindowsNoeditor/PythonAPI/examples folder and paste 
the code there. 
• Now run the code using python filename.py.  
Now the simulator runs in real time and the detections can be seen in the window. 
Make sure to install all the dependencies including OpenCV, Numpy, Pytorch and 
ultralytics before  executing the code.
