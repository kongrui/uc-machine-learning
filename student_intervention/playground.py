#!/usr/bin/env python

import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.mini.csv")
print "Student data read successfully!"

shapes = student_data.shape
print(shapes)

n_students = shapes[0]
n_features = shapes[1] - 1

# TODO: Calculate passing students
type(student_data[student_data['passed']=='yes'])
n_passed = student_data[student_data['passed']=='yes']['passed'].count()

# TODO: Calculate failing students
n_failed = student_data[student_data['passed']=='no']['passed'].count()

# TODO: Calculate graduation rate
grad_rate = np.float32(n_passed)/np.float32(n_students) * 100

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

for col_data in student_data.iteritems():
    print('===========================')
    print(col_data)
    print(type(col_data))
