import numpy as np
import math

class classifier:
    def __init__(self, vector=None, threshold=None, invert=1, debug=None):#haar_num=None, data_length=None, threshold=None, debug=None):

        # Initialize the variables
        self.thresh = threshold

        if vector is None:
            self.vector = vector
        else:
            self.vector = np.array(vector)
        self.debug = debug
        self.invert = invert

    def classify(self, example):
        '''
        Classify a single example.
        '''

        # Test if the classifier has been initialized
        if not (self.test_initialized()):
            print "Uninitialized classifier"
            return 0

        answer = np.dot(self.vector, example)

        if self.debug:
            print "Classification:", answer

        if answer < self.thresh:
            return -1 * self.invert
        else:
            return 1 * self.invert

    def weighted_error(self, dataset, actual_values, weights):
        '''
        Compute the error for the dataset with  respect to a given set of weights.
        '''

        # Test if the number of training examples, their actual values and the training weights are all the same
        if not (len(dataset) == len(actual_values) == len(weights)):
            print "Incorrect dimensions for dataset, actual_values or weights"
            return 0

        # Test if the classifier has been initialized
        if not (self.test_initialized()):
            print "Uninitailized classifier"
            return 0

        # Check whether each training example is correctly classified and update the error by the corresponding weight if incorrectly classified
        err = 0
        for i, d in enumerate(dataset):

            # Check whether the data dimensions are correct
            if (len(d) != len(self.vector)):
                print "Incorrect dimensions for training example", i
                continue

            err += weights[i] * int(actual_values[i] != self.classify(d))

            # Debugging printing
            if self.debug:
                self.debug = 0
                if actual_values[i] == self.classify(d):
                    print "Example", i, "correctly classified as", self.classify(d)
                else:
                    print "Example", i, "incorrectly classified as " + str(self.classify(d)) + ". Increasing error by:", weights[i] * int(actual_values[i] != self.classify(d))
                self.debug = 1

        if self.debug:
            print "Final error:", err

        return err

    def get_classification_vector(self):
        return self.vector

    def __str__(self):
        return "Threshold: " + str(self.thresh) + "; Invert: " + str(self.invert) + "; Vector: " + str(self.vector)

    def save(self):
        '''
        Returns a string that can be passed to the load function to recover the classifier's specifications.
        '''
        return str(self.thresh) + ";" + np.array(self.vector).tostring() + ";" + str(self.invert) + ";" + str(self.debug)

    def load(self, string):
        '''
        Resets the object's variables to what is represented in the string
        '''
        try:
            vs = string.split(";")
            t = float(vs[0])
            i = int(vs[2])
            d = int(vs[3])
            v = np.fromstring(vs[1])
        except:
            print "Unable to recover object."
            return False
        self.thresh = t
        self.invert = i
        self.debug = d
        self.vector = v
        return True

    def test_initialized(self):
        return (self.thresh != None) and (type(self.vector) != None)

    def copy(self):
        '''
        Returns a classifier object with the same attributes as the current classifier.
        '''
        return classifier(self.vector, self.thresh, self.invert, self.debug)

    def __repr__(self):
        return self.__str__()
       

def compute_haar_vector(length, haar_num):
    '''
    Computes a Haar vector of the given order for the given length
    '''
    vec = np.zeros(length)
    val = -1
    test_val = length / (2 ** (haar_num))
    if test_val > 0:
        for i in range(length):
            if i % test_val == 0:
                val *= -1
            vec[i] = val
    return vec

def compute_random_normal(length):
    vec = np.random.randn(length)
    square = math.sqrt(np.sum(vec ** 2.0))
    return (vec / square)

if __name__ == '__main__':
    a = compute_random_normal(10)
    b = compute_random_normal(10)
    aa = classifier(vector=a, threshold=0.5, invert=1, debug=0)
    bb = classifier(vector=b, threshold=0.5, invert=1, debug=0)
    print aa.vector
    print bb.vector
    print np.dot(a, b)
