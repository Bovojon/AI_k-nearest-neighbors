import sys
import random
import math
from operator import itemgetter
import operator
from sklearn.metrics import confusion_matrix

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def euclideanDistance(instance_1, instance_2, length):
    distance = 0
    for i in range(length):
        i += 1
        if is_number(instance_1[i]):
            distance += (float(instance_2[i]) - float(instance_1[i])) ** 2
        else:
            distance += instance_1[i] != instance_2[i]
    distance = math.sqrt(distance)
    return distance

def hammingDistance(instance_1, instance_2, length):
    distance = 0
    for i in range(length):
        i += 1
        if instance_1[i] != instance_2[i]:
            distance += 1
    return distance

def getArgs():
    hamming = False;
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("Usage: python main.py filePath kValue training_set random_seed")
        exit(-1)
    try:
        inFile = open(sys.argv[1], "r")
        kValue = int(sys.argv[2])
        trainingPerc = float(sys.argv[3])
        randomSeed = int(sys.argv[4])
        if len(sys.argv) == 6:
            if sys.argv[5] == "H":
                hamming = True
    except IOError:
        print("Error opening the file " + sys.argv[1])
        exit(-1)

    if int(sys.argv[2]) < 1:
        print("k value too small!")
        exit(-1)

    if float(sys.argv[3]) < 0 or float(sys.argv[3]) > 1:
        print("Training set percentage is invalid!")
        exit(-1)

    return inFile, kValue, trainingPerc, randomSeed, hamming

def processFile(inFile, randomSeed, trainingPerc):
    random.seed(randomSeed)
    training = []
    test = []
    next(inFile)
    for line in inFile:
        line = line.rstrip()
        if random.random() < trainingPerc:
            training.append(line.split(","))
        else:
            test.append(line.split(","))
    return training, test

def getNeighbors(instance, training, kValue, hamming):
    distances = []
    length = len(instance) - 1
    for node in training:
        if not hamming:
            distance = euclideanDistance(instance, node, length)
        else:
            distance = hammingDistance(instance, node, length)
        distances.append((node, distance))
    distances.sort(key=itemgetter(1))
    neighbors = []
    for i in range(kValue):
        neighbors.append(distances[i][0])
    return neighbors

def classify(neighbors):
    classifications = {}
    for node in neighbors:
        if node[0] in classifications:
            classifications[node[0]] += 1
        else:
            classifications[node[0]] = 1
    return max(classifications.items(), key=operator.itemgetter(1))[0]

def main():
    inFile, kValue, trainingPerc, randomSeed, hamming = getArgs()
    training, test = processFile(inFile, randomSeed, trainingPerc)
    correct = 0
    total = 0
    predictions = []
    true = []
    labels = []
    for i in training + test:
        labels.append(i[0])
    labels = list(set(labels))
    labels.sort()

    for node in test:
        pred = classify(getNeighbors(node, training, kValue, hamming))
        predictions.append(pred)
        true.append(node[0])
        if pred == node[0]:
            correct += 1
        total += 1
    print("Accuracy: ", correct / total)
    cf_matrix = confusion_matrix(true, predictions, labels=labels)
    print(cf_matrix)

    fout = open("Output/results_" + inFile.name.split(".")[0].split("/")[-1] + "_" + str(kValue) +"_" + str(randomSeed) + ".csv", "w")
    fout.write(",".join(labels))
    fout.write("\n")
    for i in range(len(cf_matrix)):
        for j in cf_matrix[i]:
            fout.write(str(j) + ",")
        fout.write(labels[i])
        fout.write("\n")


if __name__ == "__main__":
    main()