import math
import numpy as np  
import operator  
import pickle
import time
import heapq
import pdb

SET_SIZE = 500
K = 10

def load():
    with open("dataset/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

## priority queue API (implemented through a min heap queue)
class NearestNeighbors:
    '''
    this class is used to store the k nearest neighbors to a test image
        - storage:
        - note: we use a minheap to store the neighbors (practically, this is a dynamic array)
        - the furthest neighbor to the image is stored at the root of the heap
        - if the current element is closer than the root element (furthest):
            - push the current element into the heap
            - pop the root element off the heap
    '''
    def __init__(self):
        self.heap = []

    def add_neighbor(self, distance, neighbor_label):
        if len(self.heap) < K:
            heapq.heappush(self.heap, (-distance, neighbor_label))        
        else:
            if -self.heap[0][0] > distance:
                heapq.heappushpop(self.heap, (-distance, neighbor_label))

    def get_consensus(self):
        labels = [neighbor[1] for neighbor in self.heap]
        label_counts = {}

        # use a dictionary to count the number of times each label appears in the heap
        for label in labels:
            # default count to 0 if label not in dictionary
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # iterate through label_counts keys and call label_counts.get(key)
        # return the key with the max count
        return max(label_counts, key=label_counts.get)

    def calculate_distance(self, x1, x2, mode="L2"):
        if mode == "L1":
            return np.sum(np.abs(x1 - x2))
        else:
            print("using L2 distance function...")
            return np.linalg.norm(x1 - x2)

def kNNClassify(test_set, training_set, training_labels):    
    result = []

    mode = "L1" # this can be set to "L1" or "L2"
    print(f"using {mode} distance function...")

    for test_img in test_set:
        neighbors = NearestNeighbors()
        for i, train_img in enumerate(training_set):
            distance = neighbors.calculate_distance(test_img, train_img, mode)
            neighbors.add_neighbor(distance, training_labels[i])

        result.append(neighbors.get_consensus())
    return result

start_time = time.time()

outputlabels=kNNClassify(x_test[0:SET_SIZE],x_train,y_train)

# calculate the classification accuracy by comparing the output labels with ground truth
result = y_test[0:SET_SIZE] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s     ---" %result)

print ("---execution time: %s seconds ---" % (time.time() - start_time))
