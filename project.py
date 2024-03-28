import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import math
import random
from PIL import Image
import adaboost as ada


def lslpf(x, m, time_step, time_width):
    '''

    '''
    x = np.array(x).flatten()
    num_step = int((x.shape[0] - time_width) / time_step) + 1
    a = np.zeros((m, num_step))
    Pm = np.zeros((num_step))
    print num_step, a.shape, Pm.shape

    for i in xrange(num_step):
        print i
        t1 = int(i * time_step)
        t2 = int(t1 + time_width)
        xFrame = x[t1:t2]
        xFrame = 2 * np.finfo(float).eps * np.random.randn(len(xFrame)) + xFrame;
        #print "t1", t1
        #print 't2', t2
        #print 'shape_xframe', xFrame.shape
        #print 'xframe', xFrame
        #print '************************'

        aFrame, PmFrame = lslp(xFrame, m)
        #print aFrame
        #print PmFrame
        #print "************************"
        a[:, i] = aFrame
        Pm[i] = PmFrame

    return a, Pm

def lslp(x, m):
    '''

    '''
    x = np.array(x).flatten()
    x = x - np.mean(x)
    n = len(x)

    X = np.zeros((n-m, m))
    for i in xrange(m):
        i1 = m - i - 1
        i2 = n - i - 1
        X[:, i] = x[i1:i2];

    b = x[m:n]
    try:
        a = np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), np.matmul(X.transpose(), b))
    except:
        a = np.matmul(np.linalg.pinv(np.matmul(X.transpose(), X)), np.matmul(X.transpose(), b))

    r = np.matmul(X, a) - b
    Pm = np.sum(r ** 2) / (2 * (n-m))
    return a, Pm

def pspectrum(f, fs, a, Pm):
    f = np.array(f).flatten()
    a = np.array(a).flatten()
    w = 2 * math.pi * f
    m = len(a)
    dt = 1.0/fs

    P = (Pm * dt) / abs(1 - np.matmul(np.exp(-1j * (np.matlib.repmat(w.reshape((len(w), 1)), 1, m) * np.matlib.repmat(np.arange(1, m+1), len(w.flatten()), 1)) * dt), a)) ** 2

    return P

def pspectrumf(f, fs, a, Pm):
    numSpectra = len(Pm)
    P = np.zeros((numSpectra, len(f)))
    for i in xrange(numSpectra):
        P[i, :] = pspectrum(f, fs, a[:, i], Pm[i])
    return P

def get_wav_info(wav_file):
    '''
    Reads the data from a .wav file and combines it into a single channel.
    wav_file - The name of the .wav file
    returns - The sampling rate of the data and an array containing the single sampled values
    '''
    rate, data = wavfile.read(wav_file)
    single_channel = np.divide(np.sum(data, axis=1), data.shape[1])
    return rate, single_channel

def mtf(midi_vals):
    f = 27.5*2**((midi_vals-21)/12.0)
    return f
    

def compute_spectrogram(t1, t2, m, data, fs, tStep, tWidth):
    data = data[int(t1*fs):int(t2*fs+1)]
    print "Doing lslpf"
    a, Pm = lslpf(data, m, round(tStep), round(tWidth))
    fRange = mtf(np.arange(50, 105, 0.25))
    print "Doing pspectrumf"
    P = pspectrumf(fRange, fs, a, Pm)
    print P.shape
    

def readArrays(filename_list):
    '''
    Reads the arrays from file, normalizes them and returns a list of all of the normalized arrays.
    '''
    arr_list = []
    max_val = None
    min_val = None
    ind = 0

    # Read all of the spectrogram arrays into a single list
    for fn in filename_list:
        arr_list.append(np.log10(np.genfromtxt(fn)))
        if max_val == None:
            max_val = np.max(arr_list[ind])
            min_val = np.min(arr_list[ind])
        else:
            max_val = max(max_val, np.max(arr_list[ind]))
            min_val = min(min_val, np.min(arr_list[ind]))
        ind += 1

    # Normalize all of the arrays
    new_max = None
    new_min = None
    for i in xrange(len(arr_list)):
        arr_list[i] = (arr_list[i] - min_val)/(max_val - min_val)
        if new_max == None:
            new_max = np.max(arr_list[i])
            new_min = np.min(arr_list[i])
        else:
            new_max = max(new_max, np.max(arr_list[i]))
            new_min = min(new_min, np.min(arr_list[i]))

    return arr_list

def readOne(filename):
    arr = np.log10(np.genfromtxt(filename))
    max_val = np.max(arr)
    min_val = np.min(arr)
    arr = 2.0*(arr - min_val)/(max_val - min_val) - 1
    print np.min(arr), np.max(arr)
    return arr

def remove_one(data, pos):
    hold_data = data[:pos] + data[pos+1:]
    training_data = []
    for i in xrange(len(hold_data)):
        training_data += hold_data[i]
    return training_data

def remove_list(data, pos):
    training_data = []
    for i in xrange(len(data)):
        if not i in pos:
            training_data += data[i]
    return training_data

def get_list(data, pos):
    training_data = []
    for i in xrange(len(data)):
        if i in pos:
            training_data += data[i]
    return training_data

def k_means(positives, folds, negatives, num_class, iterations, fname):
    # Divide the data into the different folds
    fold_data = []
    fold_labels = []
    fold_count = []
    for i in xrange(len(np.unique(folds))):
        fold_data.append([])
        fold_labels.append([])
        fold_count.append(0)
    for i in folds:
        fold_count[i] += 1
    for i, p in enumerate(positives):
        f = folds[i]
        fold_data[f].append(p.flatten())
        fold_labels[f].append(1)
        fold_data[f].append(negatives[i].flatten())
        fold_labels[f].append(-1)
    for hold in fold_labels:
        print len(hold), " ", 
    print

    # Run the adaboost algorithm using cross validation
    for i in xrange(3, len(fold_data)/4):
        file_name = fname + '_output_' + str(i) + '.txt'
        open(file_name, 'w').close()
        training_data = remove_list(fold_data, np.arange(i*4, i*4+4).tolist())
        training_labels = remove_list(fold_labels, np.arange(i*4, i*4+4).tolist())

        trainer = ada.AdaBoost(num_classifiers=num_class,
                               train_iterations=iterations,
                               debug=0)
        trainer.set_data(data=training_data,
                         labels=training_labels)
        with open(file_name, 'a') as ff:
            ff.write("Training for fold " + str(i) + " with data size " + str(len(training_data)) + "\n")
        print "Training for fold " + str(i) + " with data size " + str(len(training_data))
        trainer.train(get_list(fold_data, np.arange(i*4, i*4+4).tolist()), get_list(fold_labels, np.arange(i*4, i*4+4).tolist()), file_name)
        if not trainer.save(fname + '_adaboost_classifier_fold_' + str(i) + '.txt'):
            print "Unable to save."

def get_chunks(data, chunk_size):
    chunks = []
    for i in xrange(data.shape[1]/chunk_size):
        chunks.append(data[:, (i*chunk_size):(i*chunk_size + chunk_size)])
    return chunks

def compute_labeled_dataset(datafile, label_file, window_size, name, chunk_size, save_file):

    # Read the spectrogram data from the array and clear the saving file
    data = readOne(datafile)
    open(save_file, 'w').close()

    # Create lists to store the positive and negative datapoints
    positives = []
    fold = []
    negatives = []

    # Create variables to know the start of a negative data section and to keep track of the total number of negative samples
    neg_start = 0
    total_neg = 0

    # Read through the file specifying the labeled data and create positive and negative samples from it
    with open(label_file, 'r') as f:
        count = 0
        for line in f:
            ll = line.split()

            # Find the starting and ending point of the consecutive piece of positive data
            pos_start = int(float(ll[0].strip()) / window_size)
            pos_end = int(float(ll[1].strip()) / window_size)

            # Find all the negative data samples that preceded the positive data, divide them into smaller chunks and save the data
            neg_example = get_chunks(data[:, neg_start:pos_start], chunk_size)
            for n in neg_example:
                negatives.append(n)
                np.save(name + "/" + name + "_neg_" + str(total_neg), negatives[-1])
                with open(save_file, 'a') as sav:
                    sav.write(name + "/" + name + "_neg_" + str(total_neg) + " 0 -1\n")
                total_neg += 1
            neg_start = pos_end

            # Find all the positive data samples in this consecutive piece, divide them into smaller chunks and save the data
            pos_example = get_chunks(data[:, pos_start:pos_end], chunk_size)
            for i, p in enumerate(pos_example):
                positives.append(p)
                fold.append(count)
                np.save(name + "/" + name + "_pos_" + str(count) + "_" + str(i), positives[-1])
                with open(save_file, 'a') as sav:
                    sav.write(name + "/" + name + "_pos_" + str(count) + "_" + str(i) + " " + str(count) + " 1\n")
            count += 1


    neg_example = get_chunks(data[:, neg_start:], chunk_size)
    for n in neg_example:
        negatives.append(n)
        np.save(name + "/" + name + "_neg_" + str(total_neg), negatives[-1])
        with open(save_file, 'a') as sav:
            sav.write(name + "/" + name + "_neg_" + str(total_neg) + " 0 -1\n")
        total_neg += 1

    pos_length = 0
    print "Folds:", len(fold), count
    print "Positives:"
    print len(positives)
    for p in positives:
        pos_length += p.shape[1]
    print "Positive length:", pos_length

    neg_length = 0
    print "Negatives:"
    print len(negatives)
    for n in negatives:
        neg_length += n.shape[1]
    print "Negative length:", neg_length

    print "Total length:", (pos_length + neg_length)
    print data.shape

def run_from_file(file_name, dataname):
    pos = []
    neg = []
    folds = []
    with open(file_name, 'r') as f:
        for l in f:
            line = l.strip().split()
            label = int(line[2])
            fold = int(line[1])
            data = np.load(line[0] + ".npy").flatten()
            if (label == 1):
                pos.append(data)
                folds.append(fold)
            elif (label == -1):
                neg.append(data)
            else:
                print "Problem 1"
    test_0 = neg[0]
    random.shuffle(neg)
    neg = neg[:len(pos)]
    test_1 = neg[0]
    k_means(pos, folds, neg, 10, 50, dataname)
    

def run_from_file_average(file_name, dataname):
    pos = []
    neg = []
    folds = []
    with open(file_name, 'r') as f:
        for l in f:
            line = l.strip().split()
            label = int(line[2])
            fold = int(line[1])
            data = np.mean(np.load(line[0] + ".npy"), 1).flatten()
            if (label == 1):
                pos.append(data)
                folds.append(fold)
            elif (label == -1):
                neg.append(data)
            else:
                print "Problem 1"
    test_0 = neg[0]
    random.shuffle(neg)
    neg = neg[:len(pos)]
    test_1 = neg[0]
    k_means(pos, folds, neg, 10, 50, dataname)

def run_from_file_mean(file_name, dataname):
    pos = []
    neg = []
    folds = []
    total = 0.0
    t_length = 0
    with open(file_name, 'r') as f:
        for l in f:
            line = l.strip().split()
            label = int(line[2])
            fold = int(line[1])
            loaded = np.load(line[0] + ".npy").flatten()
            total += np.sum(loaded)
            t_length += len(loaded)
            if (label == 1):
                pos.append(loaded)
                folds.append(fold)
            elif (label == -1):
                neg.append(loaded)
            else:
                print "Problem 1"
    for p in pos:
        p -= (total/t_length)
    for n in neg:
        n -= (total/t_length)
    test_0 = neg[0]
    random.shuffle(neg)
    neg = neg[:len(pos)]
    test_1 = neg[0]
    k_means(pos, folds, neg, 10, 50, dataname)

def view_data(pos, neg):
    new_pos = pos[0].reshape(len(pos[0]), 1)
    for i in xrange(1, len(pos)):
        new_pos = np.concatenate((new_pos, pos[i].reshape(len(pos[i]), 1)), axis=1)
    new_pos = ((new_pos + 1)/2 * 255).astype(np.uint8)
    print new_pos.shape
    im = Image.fromarray(new_pos)
    im.show()
    exit()

def view_data_folds(fold, savename):
    print fold[0][0].shape
    new_pos = np.ones((len(fold[0][0]), 1)) * -1
    for f in fold:
        for i in f:
            new_pos = np.concatenate((new_pos, i.reshape(len(i), 1)), axis=1)
        new_pos = np.concatenate((new_pos, np.ones((len(i), 1)) * -1), axis=1)
    new_pos = np.flip(((new_pos + 1)/2 * 255).astype(np.uint8), 0)
    im = Image.fromarray(new_pos)
    im.show()

def view_data_labeled(fold, labels, savename):
    print fold[0][0].shape
    new_pos = np.ones((len(fold[0][0]), 3, 3)) * -1
    for f in fold:
        for i in f:
            new_pos = np.concatenate((new_pos, np.dstack((i.reshape(len(i), 1), i.reshape(len(i), 1), i.reshape(len(i), 1)))), axis=1)
        new_pos = np.concatenate((new_pos, np.ones((len(i), 3, 3)) * -1), axis=1)
    print new_pos.shape
    colours = np.zeros((5, new_pos.shape[1], 3))
    count = 3
    for l in labels:
        for i in l:
            if i == 1:
                for k in xrange(colours.shape[0]):
                    colours[k][count][1] = 255
            else:
                for k in xrange(colours.shape[0]):
                    colours[k][count][0] = 255
            count += 1
        count += 3

    new_pos = np.flip(((new_pos + 1)/2 * 255).astype(np.uint8), 0)
    new_pos = np.concatenate((new_pos, colours), axis=0).astype(np.uint8)
    im = Image.fromarray(new_pos)
    im.save(savename + "_labeled.png")

if __name__ == '__main__':
    #compute_labeled_dataset('Data/P-full.txt', 'lines.txt', 0.01, 'lines', 10, 'lines_labeled.txt')
    #k_means(pos, folds, negs, 100, 100)
    #compute_labeled_dataset('Data/P-full.txt', 'cells.txt', 0.01, 'cells', 10, 'cells_labeled.txt')

    run_from_file('cells_labeled.txt', 'scaled_cells')
    run_from_file_average('cells_labeled.txt', 'scaled_cells_avg')
    run_from_file_mean('cells_labeled.txt', 'scaled_cells_mean_sub')

    run_from_file('lines_labeled.txt', 'scaled_lines')
    run_from_file_average('lines_labeled.txt', 'scaled_lines_avg')
    run_from_file_mean('lines_labeled.txt', 'scaled_lines_mean_sub')
