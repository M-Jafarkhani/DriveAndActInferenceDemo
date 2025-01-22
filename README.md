### Overview 
This project implements a video activity recognition system using the Inflated 3D Convolutional Network (I3D) model. The model is pretrained on the Kinetics dataset and fine-tuned on the Drive&Act dataset to recognize fine-grained driver activities, such as "drinking," "fastening seatbelt," and "working on a laptop."  

The system processes a video frame-by-frame, buffers frames for temporal context, makes predictions using the I3D model, and optionally saves or visualizes annotated frames with activity predictions and confidence scores.  

##### I3D model
The I3D model processes the input frames in two dimensions simultaneously:   

Spatial Dimension (Height and Width):   

1. It looks at patterns within each frame (e.g., objects, hand positions, body poses).  
2. This is done using 2D convolutions across the spatial dimensions (like in image classification models).   

Temporal Dimension (Time):  
1. It captures patterns across consecutive frames (e.g., motion, interactions).  
2. This is done using 3D convolutions, which extend regular 2D convolutions to include the time dimension.  

##### How It Works
1. Pretrained I3D Model:
The I3D model is designed for spatio-temporal activity recognition.  
It uses 3D convolutions to capture motion and temporal dynamics across video frames.
Pretrained on Kinetics and fine-tuned on Drive&Act for driver behavior recognition.  

2. Pipeline:  
Load Model: The pretrained I3D model and class mappings are loaded.  
Read Video: The input video is processed frame-by-frame.  
Buffer Frames: A rolling window of frames (e.g., 32) is maintained for temporal predictions.  
Run Predictions: The buffered frames are passed to the I3D model to predict activities.  
Annotate Frames: Each frame is overlaid with predictions (e.g., activity and confidence score).  
Save/Visualize: Annotated frames are saved to an output video or displayed.    

#### Prequisites  
pip install torch torchvision opencv-python numpy  
pip install pillow  

#### Execution
python InferenceOnSingleVideoMini.py  
The script processes a video and generates predictions.

