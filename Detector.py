import argparse
import cv2 as cv
import sys
import numpy as np
import os.path
import time

class Detector:
    def __init__(self, ):
        # Default settings values
        self.confThreshold = 0.5
        self.nmsThreshold = 0.5
        self.inpWidth  = 320
        self.inpHeight = 320
        self.args = None
        self.classesFile = ""
        self.modelConfiguration = ""
        self.modelWeights = ""
        self.classes = None
        self.net = None
        self.faces = []

    def set_settings(self, settings):
        # Initialize the parameters
        self.confThreshold      = settings["confThreshold"]
        self.nmsThreshold       = settings["nmsThreshold"]
        self.inpWidth           = settings["inpWidth"]
        self.inpHeight          = settings["inpHeight"]
        self.classesFile        = settings["classesFilePath"]
        self.modelConfiguration = settings["modelConfigurationPath"]
        self.modelWeights       = settings["modelWeights"]


        parser = argparse.ArgumentParser(description='Real Time Face Anonymization')
        parser.add_argument('--image', help='Path to image file.')
        parser.add_argument('--video', help='Path to video file.')
        self.args = parser.parse_args()

        # Load names of classes
        self.classes = None
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        net = cv.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.net = net

    # Get the names of the output layers
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            #self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
            self.faces.append([top, left, height, width])

    # All faces anonymization in image
    def anonymization_image(self, frame):

        for coord in self.faces:
            image = frame[coord[0]:coord[0]+coord[2] , coord[1]:coord[1]+coord[3]]

            (h, w) = image.shape[:2]
            xSteps = np.linspace(0, w, 10 + 1, dtype="int")
            ySteps = np.linspace(0, h, 10 + 1, dtype="int")
            # loop over the blocks in both the x and y direction
            for i in range(1, len(ySteps)):
                for j in range(1, len(xSteps)):
                    # compute the starting and ending (x, y)-coordinates
                    # for the current block
                    startX = xSteps[j - 1]
                    startY = ySteps[i - 1]
                    endX = xSteps[j]
                    endY = ySteps[i]
                    # extract the ROI using NumPy array slicing, compute the
                    # mean of the ROI, and then draw a rectangle with the
                    # mean RGB values over the ROI in the original image
                    roi = image[startY:endY, startX:endX]
                    (B, G, R) = [int(x) for x in cv.mean(roi)[:3]]
                    cv.rectangle(image, (startX, startY), (endX, endY),
                                  (B, G, R), -1)
            frame[coord[0]:coord[0]+coord[2] , coord[1]:coord[1]+coord[3]] = image

    def detect(self,):
        # Process inputs
        winName = 'Deep learning object detection in OpenCV'
        cv.namedWindow(winName, cv.WINDOW_NORMAL)

        outputFile = "yolo_out_py.avi"
        if (self.args.image):
            # Open the image file
            if not os.path.isfile(self.args.image):
                print("Input image file ", self.args.image, " doesn't exist")
                sys.exit(1)
            cap = cv.VideoCapture(self.args.image)
            outputFile = self.args.image[:-4] + '_yolo_out_py.jpg'
        elif (self.args.video):
            # Open the video file
            if not os.path.isfile(self.args.video):
                print("Input video file ", self.args.video, " doesn't exist")
                sys.exit(1)
            cap = cv.VideoCapture(self.args.video)
            outputFile = self.args.video[:-4] + '_yolo_out_py.avi'
        else:
            # Webcam input
            cap = cv.VideoCapture(2)

        # Get the video writer initialized to save the output video
        if (not self.args.image):
            vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20,
                                        (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                         round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0

        count = 0
        while cv.waitKey(1) < 0:

            # get frame from the video
            hasFrame, frame = cap.read()

            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                print("Output file is stored as ", outputFile)
                cv.waitKey(3000)
                # Release device
                cap.release()
                break

            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            self.net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = self.net.forward(self.getOutputsNames(self.net))

            # Remove the bounding boxes with low confidence
            self.postprocess(frame, outs)

            # Anonymization frame
            self.anonymization_image(frame)

            # time when we finish processing for this frame
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Draw FPS on frame
            cv.putText(frame, "FPS : "+str(int(fps)), (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            # Write the frame with the detection boxes
            if (self.args.image):
                cv.imwrite(outputFile, frame.astype(np.uint8))
            else:
                vid_writer.write(frame.astype(np.uint8))

            cv.imshow(winName, frame)
            self.faces.clear()

        # Close and destroy all allocated resources
        cv.destroyAllWindows()
        cap.release()
        vid_writer.release()

