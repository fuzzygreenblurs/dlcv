import math
import numpy as np  
import pickle
import time
import pdb

OVERFLOW_GUARDRAIL = 1e-8
TRAINING_EPOCHS = 10
NUM_CLASSES = 10
NUM_FEATURES = 784
BATCH_SIZE = 32
TEST_SET_SIZE = 500
INITIAL_W_SCALE_FACTOR = 0.5
LEARNING_RATE = 0.01

LAMBDA = 0.02

class LinearClassifier:
    def __init__(self):
        self.load()
        self.initialize_W()

    def initialize_W(self):
        self.W = np.random.randn(NUM_CLASSES, NUM_FEATURES) * INITIAL_W_SCALE_FACTOR        # 10x784

    def calculate_softmax_prob(self, scores):
        exp_scores = np.exp(scores - np.max(scores, axis=0, keepdims=True))
        return exp_scores / (np.sum(exp_scores, axis=0, keepdims=True) + OVERFLOW_GUARDRAIL)
    
    def forward(self):
        i = self.batch_idx
        self.batch_images = self.training_set[i: i + BATCH_SIZE]                            # [32x784]
        self.batch_labels = self.training_labels[i: i + BATCH_SIZE]                         # [32x1]
        S = self.W @ self.batch_images.T                                                    # scores matrix: 10x784 x 784x32 => [10x32]
        self.P = self.calculate_softmax_prob(S)                                             # normalized probabilities matrix: 10x32 (same as raw scores matrix)

        self.Pc = self.P[self.batch_labels, np.arange(BATCH_SIZE)]                          # probabilities of the correct class for each image in batch [32x1]
                                                                                            # note that `labels` encodes the row index for the right label for each image

        Lce = -np.log(self.Pc + OVERFLOW_GUARDRAIL)                                         # cross-entropy loss component encoding loss per image in batch [32x1]
        reg = (LAMBDA / 2) * np.sum(self.W ** 2)                                            # regularization loss component (L2) [1x1]. I use LAMBDA/2 here to simplify the derivative in backpass.
        self.batch_loss = np.mean(Lce) + reg                                                # final loss assigned to W [1x1]
    
    def backward_alt(self):
        one_hot = np.eye(NUM_CLASSES)[self.batch_labels].T                                  # [10x32]
        dL_dS  =  self.P - one_hot
        dLce_dW = (dL_dS @ self.batch_images) / BATCH_SIZE
        
        dLreg_dW   = LAMBDA * self.W
        self.dL_dW = dLce_dW + dLreg_dW

    def backward(self):               
        # please see attached notes for derivation of these expressions
        dLce_dPc   = -1 / (BATCH_SIZE * self.Pc + OVERFLOW_GUARDRAIL)                       # [32x1]                                                              
        dPc_dS     = self.calculate_J()                                                     # jacobian of softmaxed probability of correct classes w.r.t raw scores [32x10x10]
                                                                                            # note: each slice represents 1 image in batch

        dS_dW      = self.training_set[self.batch_idx : self.batch_idx + BATCH_SIZE].T      # note: we process the entire batch of images in 1 operation, 
                                                                                            # therefore, select the whole batch of imgs [784x32]
        
        dLce_dW    = np.zeros((10, 784))
        for i in range(BATCH_SIZE):
            '''
                dS_dW has dimensions [784x32] so for each image, we take the dot product of a slice of the jacobian 
                with the corresponding image vector. we then multiply the first term (scalar) with the result. 
                this is the dLce/dW for a single image. to get the total, we just sum the batch. 

                note: below, we break up gradient = A * B * C into A * (BC) in 2 steps for readability

                BC: [10x10] x [10x784]
            '''
            dS_dW_reshaped = np.tile(dS_dW[:, i].reshape(1, -1), (10, 1))
            BC = np.dot(dPc_dS[:, :, i], dS_dW_reshaped)                       
            dLce_dW += dLce_dPc[i] * BC

        dLreg_dW   = LAMBDA * self.W
        self.dL_dW = dLce_dW + dLreg_dW


    ''' this is the whole jacobian manually calculated. 
        this doesn't really work well in practice, due to divide-by-zero or explode errors!
        I will use a simpler formula (source: )
    '''
    # def calculate_J(self):
    #     '''
    #     generate a jacobian that encodes the partial derivatives of each softmaxed probability'
    #     with every raw score.

    #     we will have a CxC matrix (where C is the number of possible classes) for each image and 
    #     B (# of images in a batch) such matrices. thus, the final Jacobian tensor is BxCxC
    #     '''
    #     J = np.zeros((10, 10, BATCH_SIZE))                                                # [10x10x32]
    #     for i in range(BATCH_SIZE):
    #         P = self.P[:, i]                                                               # ith column of P (corresponding to ith image in batch) [10x1]
    #         outer = -np.outer(P, P)                                                        # populate off-diagonal elements of Jacobian for image [10x10]
    #         diagonal = P * (1-P)                                                           # element-wise multiplication of P_i (10x1) vector with the complement of itself [10x1]
    #         J[:, :, i] = outer                                                             # set the ith slice of J (corresponding to the given image) to be the outer [10x10x32]
    #         np.fill_diagonal(J[:, :, i], diagonal)

    #     return J

    def train(self):
        num_samples = self.training_set.shape[0]                                            # training set: 60000x784

        for e in range(TRAINING_EPOCHS):

            # # shuffle the data to avoid overfitting
            indices = np.arange(num_samples)
            np.random.seed(999)
            np.random.shuffle(indices)
            self.training_set = self.training_set[indices]
            self.training_labels = self.training_labels[indices]

            epoch_loss = 0
            batches = 0
            for i in range(0, num_samples, BATCH_SIZE):                                         # BATCH_SIZE = 32
                self.batch_idx = i
                self.forward()
                self.backward_alt()
                self.W -= LEARNING_RATE * self.dL_dW
                
                # print("batch loss: ", self.batch_loss)
                epoch_loss += self.batch_loss
                batches += 1
                
            print(f"epoch {e}: batches: {batches}, loss: {epoch_loss}")

    def predict(self):
        predictions = np.zeros(TEST_SET_SIZE)
        for i, test_img in enumerate(self.test_set[0:TEST_SET_SIZE]):
            scores = np.dot(self.W, test_img)
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
print (f"---classification accuracy for linear classifier on mnist: {accuracy * 100}% ---")
print ("---execution time: %s seconds ---" % (time.time() - start_time))