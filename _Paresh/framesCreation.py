import cv2
import os

vidcap = cv2.VideoCapture('test.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print ('Read a new frame: ', success)

  cv2.imwrite(os.path.join("C:/Users/Paresh P/PycharmProjects/fyp/DepictCleaning/Frames", "frame%05d.jpg" % count), image)

  count += 1