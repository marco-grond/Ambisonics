import classifier as cl
import numpy as np
import random
import matplotlib.pyplot as plt
import time

class AdaBoost:
    def __init__(self, num_classifiers=None, train_iterations=None, debug=None):
        '''
        Initialize the object
        '''

        # Variables used during training
        self.class_bag = []
        self.iterations = train_iterations
        self.num_class = num_classifiers

        # Lists of data used during training
        self.weights = None
        self.data = None
        self.labels = None

        # Variables determined during training
        self.final_classifiers = []
        self.alphas = []

        # Other variables
        self.debug = debug


    def set_data(self, data, labels):
        '''
        Set the data and ground truth values.
        '''

        # Check if the number of labels and datapoints are the same
        if len(data) != len(labels):
            print "Incorrect dimensions for data and labels"
            return False

        # Check to see if all of the training data have the same dimensions
        for i in xrange(1, len(data)):
            if (len(data[i-1]) != len(data[i])):
                print "Inconsistent dimensions for data."
                return False
            if not (labels[i] == 1 or labels[i] == -1):
                print "Invalid label for data."
                return False

        # Check to see if the data and labels have already been set
        if (type(self.data) == type(None) or type(self.weights) == type(None)):
            self.data = data
            self.labels = labels
            return True

        # Overwrite the data and labels
        else:
            self.data = data
            self.labels = labels
            print "Data and weights overwritten"
            return True

    def fill_bag(self):
        '''
        Fills the bag of classifiers.
        '''
        if type(self.data) == type(None):
            print "Cannot fill bag without any data examples"
            return False

        haar_num = 0
        data_length = len(self.data[0])
        while (data_length / (2 ** (haar_num))) > 0:
            vec = np.array(cl.compute_haar_vector(data_length, haar_num))
            self.class_bag += self.get_classifier_family(vec)
            haar_num += 1
        print "Num haar classifiers:", len(self.class_bag)

        for i in xrange(min(data_length, 500)):
            vec = np.array(cl.compute_random_normal(data_length)).flatten()
            self.class_bag += self.get_classifier_family(vec)

            if self.debug:
                print '-'*50
                print "Added classifiers for haar_num", haar_num, "to the bag."
                print "Number of classifiers in bag:", len(self.class_bag)
                print '-'*50

        print "Num classifiers:", len(self.class_bag)

        return True

    def get_classifier_family(self, vec):#haar_num):
        '''
        Compute a set of classifiers that all use the same Haar wavelet, but have different thresholds
        '''

        if type(self.data) == type(None):
            print "Cannot compute classifier family without any data examples"
            return False

        min_thresh = None
        max_thresh = None
        #haar_wavelet = np.array(cl.compute_haar_vector(len(self.data[0]), haar_num)).flatten()
        #vec = np.array(cl.compute_random_normal(length)).flatten()
        #print vec
        #print haar_wavelet

        # Apply the haar_wavelet to each of the training data examples to find the min and max values
        for i, td in enumerate(self.data):
            try:
                temp = np.array(td).flatten().reshape((1, len(self.data[0])))
            except:
                print "Different dimensions for wavelet and example", i
                continue

            val = np.dot(td, vec)#haar_wavelet)
            if min_thresh == None or max_thresh == None:
                min_thresh = val
                max_thresh = val
            else:
                min_thresh = min(val, min_thresh)
                max_thresh = max(val, max_thresh)
        
        if (min_thresh == None) or (max_thresh == None):
            print "Unable to find any minimum or maximum values for data and wavelet"
            return None
        if (min_thresh == max_thresh):
            min_thresh -= 1
            max_thresh += 1

        # Compute the different thresholds and create a classifier for each
        step_size = (max_thresh - min_thresh)/(self.num_class - 1)
        thresh_list = np.arange(min_thresh, max_thresh+step_size/2.0, step_size)
        class_list = []

        if self.debug:
            #print "Haar num:", haar_num
            print "Step size:", step_size
            #print "Haar wavelet:", haar_wavelet
            print "Min-max threshold:", min_thresh, 'to', max_thresh

        for t in thresh_list:
            class_list.append(cl.classifier(vector=vec,
                                            threshold=t,
                                            invert=1,
                                            debug=0))
            class_list.append(cl.classifier(vector=vec,
                                            threshold=t,
                                            invert=-1,
                                            debug=0))

        return class_list

    def find_classifier(self):
        '''
        Find and return the best classifier for the current weights.
        '''

        # Set up variables to store which classifier performed the best
        chosen = None
        c_error = None

        # Compute the error for each classifier and save the classifier with the lowest error
        count = 0
        for c in self.class_bag:
            #print "Testing classifier", count
            count += 1
            hold = c.copy()
            err = hold.weighted_error(self.data, self.labels, self.weights)
            #invert_term = 1
            #if err > 0.5:
                #err = 1 - err
                #hold.invert *= -1
            #    break
                
            if c_error == None or err < c_error:
                chosen = hold
                c_error = err

        if self.debug:
            print '*'*50
            print "Best classifier:", chosen
            print "Weighted error:", c_error
            print '*'*50

        return chosen, c_error
            

    def update_weights(self, chosen):
        '''
        Update the weights with respect to the chosen classifier.
        '''
        total_weights = 0
        correct = []
        if self.debug:
            print "Old weights:"
            print self.weights

        for i in xrange(len(self.weights)):
            new_weight = self.weights[i] * np.exp(-1.0 * self.alphas[-1] * self.labels[i] * chosen.classify(self.data[i]))
            correct.append(self.labels[i] == chosen.classify(self.data[i]))
            self.weights[i] = new_weight
            total_weights += new_weight
        for i in xrange(len(self.weights)):
            self.weights[i] = 1.0*self.weights[i]/total_weights

        if self.debug:
            print "New weights:"
            print self.weights
            print correct


    def train(self, test_data=[], test_labels=[], file_name='output.txt'):
        '''
        Compute a strong classifier by combining multiple weak classifiers.
        '''
        # Initialize the weights
        self.weights = [1.0/len(self.data)]*len(self.data)

        if len(self.class_bag) == 0:
            if not self.fill_bag():
                print "Unable to fill bag."
                return False

        if self.debug:
            print "Initialized weights:\n", self.weights

        # Main AdaBoost loop
        for i in xrange(self.iterations):
            t_start = time.time()
            if self.debug:
                print "Training iteration", (i + 1)

            # Get the best weak classifier
            chosen, err = self.find_classifier()

            print "Chosen classifier error:", err

            # Check if classification error is better than random guessing
            if err > 0.5:
                break

            # Save new classifier and alpha values to list
            r = 0
            for j, d in enumerate(self.data):
                r += self.weights[j] * self.labels[j] * chosen.classify(d)
            alpha = 0.5 * np.log((1.0 + r)/(1.0 - r))
            self.final_classifiers.append(chosen)
            self.alphas.append(alpha)

            # Update weights with respect to current classifier
            self.update_weights(chosen)

            if self.debug:
                print "Added:\n"
                print "\tClassifier:", chosen
                print "\tAlpha:", alpha
                print "\tError:", err
                print

            t_end = time.time()
            if (len(test_data) == len(test_labels)) and (len(test_data) > 0):
                total_correct = 0.0
                false_pos = 0.0
                false_neg = 0.0
                # Testing accuracies
                for j in xrange(len(test_data)):
                    val = self.classify(test_data[j])
                    #test_classification.append(val)
                    if val == test_labels[j]:
                        total_correct += 1
                    elif val == -1:
                        false_neg += 1
                    else:
                        false_pos += 1
                #plot_dataset(test_data, test_classification)

                train_correct = 0.0
                t_false_pos = 0.0
                t_false_neg = 0.0
                # Training accuracies
                for j in xrange(len(self.data)):
                    val = self.classify(self.data[j])
                    #test_classification.append(val)
                    if val == self.labels[j]:
                        train_correct += 1
                    elif val == -1:
                        t_false_neg += 1
                    else:
                        t_false_pos += 1
                print_str = ("Iter: " +  str(i) + "\tTrain_Acc: " + str(train_correct/len(self.data)) + "\tTrain_F_Pos: " + str(2*t_false_pos/len(self.data)) + "\tTrain_F_Neg: " + str(2*t_false_neg/len(self.data)) + "\tTest_Acc: " + str(total_correct/len(test_data)) + "\tTest_F_Pos: " + str(2*false_pos/len(test_data)) + "\tTest_F_Neg: " + str(2*false_neg/len(test_data)) + "\tTime: " + str(t_end - t_start) + "\n")
                print print_str

                with open(file_name, 'a') as f:
                    f.write(print_str)
                    #f.write("Iteration: " +  str(i) + "\tTrain_Accuracy: " + str(train_correct/len(self.data)) + "\tTest_Accuracy: " + str(total_correct/len(test_data)) + "\tFalse_Pos: " + str(false_pos/len(test_data)) + "\tFalse_Neg: " + str(false_neg/len(test_data)) + "\tTime: " + str(t_end - t_start) + "\n")

        return True

    def classify(self, data):
        '''
        Classifies the given data into one of the two classes.
        '''

        # Test to see if the classifier has been trained yet
        if (len(self.final_classifiers) == 0) or (len(self.alphas) == 0):
            print "Untrained classifier"
            return 0

        # Test to see if the data has the correct dimension
        if len(data) != len(self.final_classifiers[0].vector):
            print "Incorrect dimension for data"
            return 0

        # Classify the data
        total = 0
        for i, fc in enumerate(self.final_classifiers):
            total += self.alphas[i] * fc.classify(data)

        # Apply the sign function
        if total < 0:
            return -1
        else:
            return 1

    def save(self, file_name):
        try:
            open(file_name, 'w').close()
            with open(file_name, 'a') as f:
                for i in xrange(len(self.alphas)):
                    f.write(str(self.alphas[i]) + ":" + self.final_classifiers[i].save() + "\n")
            return True
        except:
            return False

    def load(self, file_name):
        self.alphas = []
        self.final_classifiers = []
        with open(file_name, 'r') as f:
            for l in f:
                clean = cl.classifier()
                line = l.split(":")
                try:
                    self.alphas.append(float(line[0]))
                except:
                    print "Unable to load classifier from file"
                    return False
                if clean.load(line[1]):
                    self.final_classifiers.append(clean)
                else:
                    print "Unable to load classifier from file"
                    return False
        return True


def plot_dataset(data, labels):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in xrange(len(data)):
        if labels[i] == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])
    plt.scatter(x1, y1, c='b', marker='.')
    plt.scatter(x2, y2, c='r', marker='.')
    plt.show()

if __name__ == '__main__':
    dataset =  [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 0, 0,], [0, 0, 1, 1], [1, 0, 1, 0], [-1, 1, -1, 1]]
    labels = [1, 1, 1, -1, -1, -1]
    test_1 = AdaBoost(num_classifiers=10, train_iterations=2, debug=0)
    test_1.set_data(dataset, labels)
    test_1.train()
    print test_1.final_classifiers
    print test_1.alphas
    test_1.save('test_save.txt')
    test_2 = AdaBoost()
    test_2.load('test_save.txt')
    print test_2.final_classifiers
    print test_2.alphas
