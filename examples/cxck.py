from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import *
import json
import csv
import pandas as pd
import numpy as np
from collections import defaultdict

class DataSource:
    def __init__(self, filename):
        self.filename=filename
        self.data = []
        self.data_map = None
        with open(filename, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    header = row
                    print(f'Column header: {", ".join(header)}')
                else:
                    person = {"score": float(row[1]), "class": row[4]=="True", "box": [0.,1.,0.,1.]}
                    dog = {"score": float(row[2]), "class": row[5]=="True", "box": [0.,1.,0.,1.]}
                    self.data.append({
                        'frame_id': int(row[0]),
                        'person': person,
                        'dog': dog })
                line_count += 1
        self.data_size = len(self.data)
        self.build()

    def build(self):
        persons = []
        dogs = []
        for frame in self.data:
            person_bound = Bounds3D(
                t1 = frame['frame_id'],
                t2 = frame['frame_id']+1, 
                x1 = frame['person']['box'][0], 
                x2 = frame['person']['box'][1], 
                y1 = frame['person']['box'][2], 
                y2 = frame['person']['box'][3]
            )
            person_payload = {"score": frame['person']['score'], "class": frame['person']['class']}
            person_interval = Interval(person_bound, payload = person_payload)
            persons.append(person_interval)
            
            dog_bound = Bounds3D(
                t1 = frame['frame_id'],
                t2 = frame['frame_id']+1, 
                x1 = frame['dog']['box'][0], 
                x2 = frame['dog']['box'][1], 
                y1 = frame['dog']['box'][2], 
                y2 = frame['dog']['box'][3]
            )
            dog_payload = {"score": frame['dog']['score'], "class": frame['dog']['class']}
            dog_interval = Interval(dog_bound, payload = dog_payload)
            dogs.append(dog_interval)

        self.data_map = {
                'person': IntervalSetMapping({0: IntervalSet(persons)}),
                'dog': IntervalSetMapping({0: IntervalSet(dogs)})
        }

    def getall(self):
        return self.data_map

    def get(self, key):
        return self.getall()[key]

ds = DataSource("../../MMVP/data/result.csv")
bikes = ds.get("dog").filter(lambda interval: interval['payload']['class'] == True)
person = ds.get("person").filter(lambda interval: interval['payload']['class'] == True)

print(f'len(person) = {len(person[0])}, len(bikes) = {len(bikes[0])}')

import time
t = time.time()
person_intersect_bike = person.join(
    bikes,
    predicate = Bounds3D.T(equal()),
    merge_op = lambda interval1, interval2: Interval(
        interval1['bounds'].span(interval2['bounds']),
        payload = {"person": interval1['payload'], "bike":interval2['payload']}
    ),
    window = 0.0,
    progress_bar = True
)
t = time.time() - t

print("Time cost: ", t)
print("number of results: ", len(person_intersect_bike[0]))
