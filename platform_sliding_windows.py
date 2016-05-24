'''Detect a the landing platform if present in the image'''

import cv2
import numpy as np
from platform_detector.detector import platform_detector, bow_features
from platform_detector.pyramid import pyramid
from platform_detector.non_maximum import non_max_suppression_fast as nms
from platform_detector.sliding_window import sliding_window



def in_range(number, test, thresh=0.2):
  return abs(number - test) < thresh


# initialise camera link
camera_port = 1
camera = cv2.VideoCapture(camera_port)

svm, extractor = platform_detector()
input("Press Enter to continue...")

detect = cv2.xfeatures2d.SIFT_create()

# Sliding window size
w, h = 60, 40

# keep looping
while True:
  img = camera.read()

  rectangles = []
  counter = 1
  scaleFactor = 1.25
  scale = 1
  font = cv2.FONT_HERSHEY_PLAIN

  for resized in pyramid(img, scaleFactor):  
    scale = float(img.shape[1]) / float(resized.shape[1])
    for (x, y, roi) in sliding_window(resized, 20, (w, h)):
    
      if roi.shape[1] != w or roi.shape[0] != h:
        continue

      try:
        bf = bow_features(roi, extractor, detect)
        _, result = svm.predict(bf)
        a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        print "Class: %d, Score: %f" % (result[0][0], res[0][0])
        score = res[0][0]
        if result[0][0] == 1:
          if score < -1.0:
            rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale)
            rectangles.append([rx, ry, rx2, ry2, abs(score)])
      except:
        pass

      counter += 1

  windows = np.array(rectangles)
  boxes = nms(windows, 0.25)


  for (x, y, x2, y2, score) in boxes:
    print x, y, x2, y2, score
    cv2.rectangle(img, (int(x),int(y)),(int(x2), int(y2)),(0, 255, 0), 1)
    cv2.putText(img, "%f" % score, (int(x),int(y)), font, 1, (0, 255, 0))

  cv2.imshow("img", img)

  key = cv2.waitKey(1) & 0xFF

  # if the 'q' key is pressed, stop the loop
  if key == ord("q"):
    break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
