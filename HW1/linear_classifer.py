import math
import numpy as np  
import pickle
import time

NUM_CLASSES = 10
NUM_FEATURES = 784
NUM_TRIALS = 100
TEST_SET_SIZE = 500
INITIAL_W_SCALE_FACTOR = 0.01

class LinearClassifier:
    def __init__(self):
        self.load()
        self.generate_randomized_W_matrices()

    def generate_randomized_W_matrices(self):
        # weights tend to start with small values (so we multiply by 0.01)
        self.Ws = np.random.randn(NUM_TRIALS, NUM_CLASSES, NUM_FEATURES) * INITIAL_W_SCALE_FACTOR

    def calculate_softmax_prob(self, scores):
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    def train(self):
        # TODO: is there a more efficient way to batch process scores?
        self.scores = np.zeros((NUM_TRIALS, NUM_CLASSES, self.num_images))
        for i, W in enumerate(self.Ws):
            # recall training set is 60000 x 784. training_set.T is 784 x 60000
            # scores: 100 x (10 x 784 * 784 x 60000) = 100 x (10 x 60000)
            self.scores[i] = np.dot(W, self.training_set.T)
        
        losses = np.zeros(NUM_TRIALS)
        for i in range(NUM_TRIALS):
            # assuming that the training_labels are one-hot encoded (1 for correct label, 0 for all others),
            # we can simplify the loss function to simply be -log(probability of correct label)
            softmax_probs = self.calculate_softmax_prob(self.scores[i])

            ''' note: we can efficiently find the correct class probabilities for each image
                - np.arange(self.num_images) provides the column indices for all images.
                - the value of self.training_labels provides the correct class index.
                
                - thus, over 60K images, training_labels[i] will resolve to the row index 
                of correct class for each image, in softmax_probs.
                
                - similarly, np.arange(self.num_images) will resolve to the column index of each 
                of the 60k images.
            '''
            correct_class_probs = softmax_probs[self.training_labels, np.arange(self.num_images)]
            log_probs = -np.log(correct_class_probs)
            losses[i] = np.mean(log_probs)

        self.optimal_W = self.Ws[np.argmin(losses)]

    def predict(self):
        predictions = np.zeros(TEST_SET_SIZE)
        for i, test_img in enumerate(self.test_set[0:TEST_SET_SIZE]):
            scores = np.dot(self.optimal_W, test_img)
            predictions[i] = np.argmax(scores)

        return predictions

    def load(self):
        with open("dataset/mnist.pkl",'rb') as f:
            mnist = pickle.load(f)

        self.training_set    = mnist["training_images"]
        self.training_labels = mnist["training_labels"]
        self.test_set        = mnist["test_images"]
        self.test_labels     = mnist["test_labels"]

        self.training_set    = self.training_set.astype(float)
        self.test_set        = self.test_set.astype(float)
        self.num_images      = self.training_set.shape[0]

classifier = LinearClassifier()

start_time = time.time()
classifier.train()
result = classifier.test_labels[0:TEST_SET_SIZE] - classifier.predict()
accuracy = (1 - np.count_nonzero(result)/len(result))
print ("---classification accuracy for linear classifier on mnist: %s ---" %accuracy)
print ("---execution time: %s seconds ---" % (time.time() - start_time))