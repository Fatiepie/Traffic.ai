# Traffic.ai
Welcome to traffic.ai, an artificial intelligence system that learns and identifies the different traffic signs available for road use, using TensorFlow, Keras API, and OpenCV! The AI who I have named Francesco gets fed in with a dataset (the German Traffic Sign Recognition Benchmark or gtsrb dataset) to teach it the different types of traffic signs that are there. 

Francesco then uses the different libraries to create a convolutional neural network (or CNN for short) to train an ML model to correctly identify the said traffic signs. The CNN consists of 2 convolutions, 1 max pool, and a single hidden layer with a dropout rate of 50% to avoid overfitting.

Francesco would then train the CNN model for 10 EPOCHS or 10 reps to complete the task. I played around with the architecture and figured that it would best if I increased the number of nodes in the convolution and hidden layers to create a better model.

# Demo

https://youtu.be/y_QNbZBLoNE

## Experimentation Process

Started with a very basic neural network structure, similar to the one in the lecture:

- 1 convolutional layer, learning filters using a 3x3 kernel
- 1 max-pooling layer, using a 2x2 pool size
- 1 hidden layer with 128 nodes
- 0.5 dropout rate
- output layer with output units for all traffic sign categories

Later, I tried to validate my results with some testing and played around with the architecture to see what CNN attribute affects the network.

## Experiment Results

| #   | Modification                                                                                  | Testing accuracy |
| :-- | :-------------------------------------------------------------------------------------------- | :--------------- |
| 1   | Original Model                                                                                | `0.9610`         |
| 2   | Add second convolutional layer, identical to the first                                        | `0.9710`         |
| 3   | Add second maxpooling layer (after the second convolutional layer), with a size of 2x2        | `0.9273`         |
| 4   | Remove second maxpooling layer, increased kernal size in second convolutional layer to (5, 5) | `0.9355`         |
| 5   | Double number of filters to 128 in second convolutional layer                                 | `0.9567`         |
| 6   | Double number of filters to 64 in second convolutional layer                                  | `0.9581`         |
| 7   | Double number of nodes in hidden layer to 256                                                 | `0.9438`         |
| 8   | Add second hidden layer (both layers with 128 nodes) with no dropout                          | `0.9444`         |
| 9   | Add second hidden layer (one with 128 nodes and the other with 256 nodes)with dropout         | `0.9372`         |
| 10  | Add 3 layers with 128 nodes with dropout                                                      | `0.9555`         |
| 11  | Added 3 layers with 128 nodes with dropout and another convolution layer                      | `0.9417`         |
| 12  | Increase dropout rate to 0.7 with base model                                                  | `0.0568`         |
| 13  | Decrease dropout rate to 0.1 2 Convolutions                                                   | `0.9282`         |
| 14  | Added a secong maxpooling of size 4x4 after the conv layer                                    | `0.8316`         |
| 15  | Ran 2 Convolutions, 3 hidden layers with a dropout rate of 0.5, on 20 EPOCHS                  | `0.9579`         |

## Discussion

Used the original model and found that it was a decent model with a test accuracy of 0.9610 so ~96%. I then wanted to figure out how each attribute if the CNN - be it the EPOCHS the model trained on, or the amount of convolutions it went through - would affect the architecture. Overall, I found that the more hidden layers and the more convolutions you have, the better the model. However, this comes at a cost as the ML model would be very expensive in terms of computing costs, so you would have to wait for a while to get these models done - not hours, but a few more mins as compared to a few seconds. The max pooling layer would obviously have to increase with another convolution as it would help in compressing the processed image better. I also found that if you add more nodes to the hidden layer, you would have a decent model to adapt to many datasets. In conclusion, the model would be optimized if you decreased the dropout to 0.4, added another convolution layer with a bigger kernal size,added another max pooling layer with an equivalent kernel matrix and added another hidden layer with a bigger kernal size and more nodes.

