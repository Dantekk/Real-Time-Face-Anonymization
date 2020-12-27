# Real-Time-Face-Anonymization
Real Time Face Anonymization using OpenCV's DNN on YOLO v3 tiny pretrained weights model. </br>
After face detection, each frame is anonymized using a pixelization algorithm.

# Repository Contents
The contents of this repository are as follows: 
* *results* folder contain examples gif.
* *work_files* folder contain configuration and weights model files.
* run.py is script for run the program

# Run the program
To use *run.py* for run the program. In particulary : 
* `python3 run.py --image= img_file` for image anonymization
* `python3 run.py --video= video_file` for video anonymization
* `python3 run.py` (without parametrer) for Real Time Face anonymization using webcam

# Example Results
Output examples of video anonymization. </br>
I got around 15 FPS on average using Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz.
| Before  |  After |
:---------:|:-----:|
![](https://github.com/Dantekk/Real-Time-Face-Anonymization/blob/main/results/Emma_no_frammed_gif.gif) | ![](https://github.com/Dantekk/Real-Time-Face-Anonymization/blob/main/results/Emma_frammed_gif.gif)

# How to improve results
1. YOLO v3 tiny have fast inference time but is not much accuracted. You could use a more accuracted model (against a slower inference time).
2. Use techniques when face detection fails (for examples, you could use face tracking algorithm).

# Important Notes:
* Used Python Version: 3.6.0
* Used OpenCV Version: 4.1.2
* Used Numpy Version: 1.18.5
