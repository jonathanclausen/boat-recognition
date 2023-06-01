# importing OpenCV, time and Pandas library
import cv2, time, pandas, math
# importing datetime class from datetime library
from datetime import datetime
from datetime import timedelta
from datetime import time
import os
import numpy as np

import time as ti
imageFolderDir = r'C:\\projects\\github\\boat-recognition\\motionimages\\good-images'

def in_between(now, start, end):
    if start <= end:
        return start <= now < end
    else: # over midnight e.g., 23:30-04:15
        return start <= now or now < end

previous_frame = None
DATETIMEFORMAT = "%Y-%m-%d-%H-%M-%S-%f"

for filename in os.scandir(imageFolderDir):
    if filename.is_file():
        print(filename.path)

        if not in_between(datetime.now().time(), time(8), time(19)):
            print(f"time is {str(datetime.now().time())}. Therefore im sleeping. I'll wake up at {str(time(8))}.")
            time.sleep(10 * 60)

        original_frame = cv2.imread(filename.path)
    
        frame = original_frame[150:350, 0:640]
        
        # Initializing motion = 0(no motion)
        motion = 0
    
        # Converting color image to gray_scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Converting gray scale image to GaussianBlur 
        # so that change can be find easily
        gray = cv2.GaussianBlur(gray, (11,11), 0)
    
        # In first iteration we assign the value 
        if previous_frame is None:
            previous_frame = gray
            continue
    
        # Difference between static background 
        # and current frame(which is GaussianBlur)
        diff_frame = cv2.absdiff(previous_frame, gray)
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 3)

        # If change in between static background and
        # current frame is greater than 30 it will show white color(255)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        #thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    
        # Finding contour of moving object
        cnts,_ = cv2.findContours(thresh_frame.copy(), 
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        base_path = imageFolderDir + "\\test" + "\\" + datetime.now().strftime(DATETIMEFORMAT)
        save_orig_path = base_path + "-original.jpg"
        save_contour_path = base_path + "-contour.jpg"
        for contour in cnts:
            
            (x, y, w, h) = cv2.boundingRect(contour)
            
            scale_factor = 1.2
            cy = y+h
            cx = x+w
            contour_area = cv2.contourArea(contour)

            print("Contour: " + str(contour_area))
            if contour_area < 10000 and contour_area > 300:
                time.sleep(1)
    
                cv2.imwrite(save_orig_path, original_frame)
                print(f"Saved original image to '{save_orig_path}'")
                cv2.imwrite(save_contour_path, frame[y:cy, x:cx])
                motion = 1
                continue

            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
        previous_frame = gray
    
        # Displaying image in gray_scale
        cv2.imshow("Gray Frame", gray)
    
        # Displaying the difference in currentframe to
        # the staticframe(very first_frame)
        cv2.imshow("Difference Frame", diff_frame)
    
        # Displaying the black and white image in which if
        # intensity difference greater than 30 it will appear white
        cv2.imshow("Threshold Frame", thresh_frame)
    
        # Displaying color frame with contour of motion of object
        cv2.imshow("Color Frame", frame)
    
        

        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            # if something is movingthen it append the end time of movement
            if motion == 1:
                time.append(datetime.now())
            break

        ti.sleep(0.1)