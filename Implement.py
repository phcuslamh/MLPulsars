#######################################################
# Name: Phuc H. Lam
# NetID: plam6
# Email: plam6@u.rochester.edu
#######################################################

import numpy as np
import matplotlib.pyplot as plt
import MLP_class as nn
import time

start_time = time.time()

# Create a matrix of training data
# FOR USERS: modify training file name here.
with open('project2Data(1)/data/htru2.train') as f1:
    Lines1 = f1.readlines()
data_train = []
for line in Lines1:
    lst = [float(n) for n in line.split(' ') if n.strip()]
    data_train.append(lst)
data_train = np.array(data_train)

# Create a matrix of development data
# FOR USERS: modify testing file name here.
with open('project2Data(1)/data/htru2.dev') as f2:
    Lines2 = f2.readlines()  
data_dev = []
for line in Lines2:
    lst = [float(n) for n in line.split(' ') if n.strip()]
    data_dev.append(lst)
data_dev = np.array(data_dev)

# Implement
# FOR USERS: Modify the parameters here
num_input = len(data_train[0]) - 1
num_hidden = 500
num_output = 1
epoch = 1000
batch_size = 512
learn_rate = 0.01

# Check if batch_size is valid 
# Invalid batch_size if it is larger than the number of data points
if (batch_size > len(data_train)):
    print('Invalid batch size. Try a smaller batch size.')
    raise NotImplementedError

# Initializing the random number generator
rng = np.random.default_rng(seed = None)

# Validation set
data_dev = np.transpose(data_dev)
print(data_dev.shape)
val_num = len(data_dev)
valX = np.array(data_dev[0:(val_num - 1)])
valY = np.transpose(data_dev[val_num - 1])
valY = valY.reshape(-1, 1)
print(valX.shape)
print(valY.shape)

# Train network
mlp = nn.MLP(num_input, num_hidden, num_output)
losses = []
accuracies = []
precisions = []
recalls = []
F1_scores = []

for _ in range(epoch):
    copy_data_train = data_train
    rng.shuffle(copy_data_train)
    data = copy_data_train[0:batch_size, :]
    data = np.transpose(data)
    trainX = data[0:(len(data) - 1)]
    trainX = np.array(trainX)
    trainY = np.transpose(data[len(data) - 1])
    trainY = trainY.reshape(-1, 1)
    trainY = np.transpose(trainY)
    mlp.grad_des(trainX, trainY, learn_rate)
    losses.append(mlp.cross_entropy_err(valX, valY))

    # Compute prediction
    # If the Y value is < 0.5, then the prediction of the output is 0; otherwise it is 1.
    valY_predwhole = mlp.forward_prop(valX)
    valY_pred = valY_predwhole[0]
    valY_pred = np.array(valY_pred)
    for i in range(len(valY_pred)):
        if (valY_pred[i] < 0.5):
            valY_pred[i] = 0
        else:
            valY_pred[i] = 1
    
    # Compute accuracy
    correct_pred = 0
    for i in range(len(valY_pred)):
        if (valY_pred[i] == valY[i]):
            correct_pred += 1
    accuracy = 100 * (correct_pred / len(valY)) 
    accuracies.append(accuracy)

    # Compute precision, recall, and F1 scores (all converted to percents, so they range from 0 to 100)
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(valY_pred)):
        if (valY_pred[i] != valY[i]):
            if (valY[i] == 1):
                FN += 1
            else:
                FP += 1
        else:
            if (valY[i] == 1):
                TP += 1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 200 * P * R / (P + R)
    precisions.append(100 * P)
    recalls.append(100 * R)
    F1_scores.append(F1)

# Print run time
print("Elapsed time (in seconds): ", (time.time() - start_time))

# Print losses
losses = np.array(losses)
accuracies = np.array(accuracies)
precisions = np.array(precisions)
recalls = np.array(recalls)
F1_scores = np.array(F1_scores)
plt.figure()
Xloss = np.array(range(epoch))
plt.plot(Xloss, losses, color="red")
plt.show()

# Print accuracies, precisions, recalls, and F1 scores
plt.figure()
plt.plot(Xloss, accuracies, color="green")
plt.plot(Xloss, precisions, color="blue")
plt.plot(Xloss, recalls, color="orange")
plt.plot(Xloss, F1_scores, color="purple")
plt.show()