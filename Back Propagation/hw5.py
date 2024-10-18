import numpy
import numpy as np
from PIL import Image


with open('../../../../../github/Dsci552/Back Propagation/downgesture_train.list') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

learning_rate = 0.001
weights1 = np.random.uniform(-0.01, 0.01, (960, 1000))
weights2 = np.random.uniform(-0.01, 0.01, (1000, 1))


def sigmoid(x):
    # print(x)
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    # print(x)
    return 1 / ((1 + np.exp(x)) * (1 + np.exp(-x)))


def loss(x, y):
    return (y - x) ** 2


for x in range(100):
    for file_name in lines:
        image = Image.open(file_name)
        s = 'gestures/A/A_down_1.pgm'
        matrix = np.asarray(image) / 255.0
        # print(matrix.shape)
        flatten_matrix = matrix.flatten()
        # print(flatten_matrix)
        input_x = flatten_matrix
        # feed forward
        s1 = numpy.dot(input_x, weights1)
        # print(s1)
        x1 = sigmoid(s1) #(1, 1000)
        deri_x1 = derivative_sigmoid(x1)
        s2 = numpy.dot(x1, weights2)
        x2 = sigmoid(s2) #(1, 1)
        deri_x2 = derivative_sigmoid(x2)
        print("forward ans: ", x2)
        if 'down' in file_name:
            y = 1
        else:
            y = 0
        print('label', y)
        print("loss: ", loss(x2, y))

        # back propagation
        delta_2 = 2*(x2 - y)*deri_x2 #(1, 1)
        delta_2 = delta_2.reshape(1,1)
        x1 = x1.reshape(1000, 1)
        # print(weights2.shape)

        weights2 = weights2 - numpy.dot(x1, delta_2) * learning_rate
        # print(weights2.shape)
        deri_x1 = deri_x1.reshape(1000, 1)
        # delta_1 = weights2 dot delta12 multiply deri_x1
        delta_1 = np.dot(weights2, delta_2)
        delta_1 = delta_1 * deri_x1

        # weights1 = weights1 - lr * delta_1 * input_x
        input_x = input_x.reshape(1, 960)
        weights1 = weights1 - np.transpose(np.dot(delta_1, input_x))



with open('../../../../../github/Dsci552/Back Propagation/downgesture_test.list') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

correct = 0
count = 0
for file_name in lines:
    print(file_name)
    count += 1
    image = Image.open(file_name)
    matrix = np.asarray(image) / 255.0
    # print(matrix.shape)
    flatten_matrix = matrix.flatten()
    # print(flatten_matrix)
    input_x = flatten_matrix

    # feed forward
    s1 = numpy.dot(input_x, weights1)
    # print(s1)
    x1 = sigmoid(s1)  # (1, 1000)
    deri_x1 = derivative_sigmoid(x1)
    s2 = numpy.dot(x1, weights2)
    x2 = sigmoid(s2)  # (1, 1)
    deri_x2 = derivative_sigmoid(x2)
    print("forward ans: ", x2)
    if 'down' in file_name:
        y = 1
    else:
        y = 0
    print('label', y)
    if x2 >= 0.5 and y == 1:
        correct += 1

    if x2 < 0.5 and y == 0:
        correct += 1


print("accuracy: ", correct / count)