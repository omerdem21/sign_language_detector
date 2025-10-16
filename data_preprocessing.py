#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:34:24 2025

@author: omererdemdilek
"""

import os
import pickle

import mediapipe as mp
import cv2


mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_style=mp.solutions.drawing_styles

hands=mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR="./data"

data=[]
labels=[]

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        if not any(img_path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            print(f"Geçersiz dosya atlandı: {img_path}")
            continue

        img_path_full = os.path.join(dir_path, img_path)
        img = cv2.imread(img_path_full)

        if img is None:
            print(f"Uyarı: Görsel yüklenemedi: {img_path_full}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                data.append(data_aux)
                labels.append(dir_)



f=open("data.pickle","wb")
pickle.dump({"data":data,"labels":labels},f)
f.close()
