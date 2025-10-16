#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:55:44 2025

@author: omererdemdilek
"""

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open("data.pickle", "rb"))

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(xtrain, ytrain)

y_pred = model.predict(xtest)

score = accuracy_score(y_pred, ytest)

print("başarı: ", score * 100)

f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()
