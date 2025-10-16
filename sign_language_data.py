#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:58:02 2025

@author: omererdemdilek
"""
import os
import cv2

DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_data = 4
data_size = 100 
cap = cv2.VideoCapture(1)

# Kamera açıldı mı kontrol et
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

for i in range(number_of_data):
    class_dir = os.path.join(DATA_DIR, str(i))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        
    print(f"class of data {i}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Image could not be retrieved!")
            break

        cv2.putText(frame, "Press q to collect data", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255), 3)
        cv2.imshow("frame", frame)
      
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    counter = 0

    while counter < data_size:
        ret, frame = cap.read()
        if not ret:
            print("Image could not be retrieved!")
            break

        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()

        

  