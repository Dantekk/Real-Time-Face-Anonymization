# Usage example:  python3 Detector.py --video=video.mp4
#                 python3 Detector.py --image=image.jpg
#		  python3 Detecror.py (without parametrer, run in real time mode)

from Detector import Detector

def main():
    d = Detector()
    settings = {"confThreshold" : 0.5,
                "nmsThreshold" : 0.5,
                "inpWidth" : 320,
                "inpHeight" : 320,
                "classesFilePath" : "work_files/face.names",
                "modelConfigurationPath" : "work_files/face-yolov3-tiny.cfg",
                "modelWeights" : "work_files/face-yolov3-tiny.weights",
                }
    d.set_settings(settings)
    d.detect()

if __name__ == "__main__":
    main()
