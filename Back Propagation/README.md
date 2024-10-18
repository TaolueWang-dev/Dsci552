# Project Report
## Problem
In the directory gestures, there is a set of images1 that display "down" gestures (i.e., thumbs-down images) or other gestures. In this assignment, you are required to implement the Back Propagation algorithm for Feed Forward Neural Networks to learn the down gestures from training instances available in downgesture_train.list. The label of an image is 1 if the word "down" is in its file name; otherwise the label is 0. The pixels of an image use the gray scale ranging from 0 to 1. In your network, use one input layer, one hidden layer of size 100, and one output perceptron. Use the value 0.1 for the learning rate. For each perceptron, use the sigmoid function Ɵ(s) = 1/(1+e-s). Use 1000 training epochs; initialize all the weights randomly between -0.01 and 0.01 (you can also choose your own initialization approach, as long as it works); and then use the trained network to predict the labels for the gestures in the test images available in downgesture_test.list. For the error function, use the standard squared error. Output your predictions and accuracy.
The image file format is "pgm" <http://netpbm.sourceforge.net/doc/pgm.html>. Please follow the link for the format details. You can either use a third-party library to read these image files or easily read them yourself.
You can write your programs in any programming language. However, you will have to implement the algorithms yourself instead of using library functions (except for reading "pgm" image files). In your report, please provide a description of the data structures you use, any code-level optimizations you perform, any challenges you face, and of course, the requested outputs.
## Data Structure
The image has the pixel with size (30, 32). After reading the image file , 
I save it in matrix by using the numpy library. Then I flatten it as a vector with size(960, 1).
I use two matrix as Hidden Layer. One is (960, 1000) and another is (1000, 1). The weights for the network has been initialed randomly between -0.1 to 0.1.

## Formula Derivation
Assume neuro network has L layers.  
Error Function:
```math
e = \left(x_1^{(L)} - y\right)^2
```

