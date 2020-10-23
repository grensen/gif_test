# goodgame|one Test Release
## Play Neural Networks to the End

goodgame or shorter gg is a huge trade-off to express neural networks and work with them in a new way.
gg comes from e-sports and was the name of my team years ago, but more important, after a game the teams say gg for a good time, 
to make the long story short, gg treats neural networks like a good game.

The rules are simple, every training sample and every custom sample is trainable. The goal is to reach the highest accuracy for the untrainable test data.

---
Let me give a first example of a test:
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_hello_goodgame.gif?raw=true)
*A neural network is initiated with random weight values to create a breaking symmetry.*

If you watched the neural network series from [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), you know the 784-16-16-10 neural network already. That's important because the starting point with goodgame is this network with the default hyperparameters.
After a first MNIST training over 60000 samples the resulting test accuracy brings 93.29%, that is pretty good the best I know.

---
Let's create a sample and train it:
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_create_inputs_and_train.gif?raw=true)
*Note: 784-16-16-10 + one training = 93.29% for the test. 
The weights work with a fixed seed, so the results cannot change, even if the initialization was randomly, what we need. But if one neuron is added or removed, the whole random values will change, that makes it hard to get confidence. A good way to see this is to beat the test accuracy with a bigger network.*


Before we go further let's checkout what goodgame can show, the input neurons are red rectangles on the left and express the samples.
Every input neuron is fully connected with a weight to every output neuron on the next layer.
The weights in green chartreuse show positive values and the reds show negative values.
The highest or lowest color depends on the highest or lowest weight here and all other weights show colors between.
Note, the values are the colors in a good, but not an exact way. 
If the highest weight value will be 1.01 on this layer, the range can be really big and values with 0.07 and 0.04 can show the same color.

The neurons show their connections only if they are ReLU activated, if the value was below 0 the neuron is set to 0.
Here we can get a good feeling how the weights works, the highest neuron color represents also the highest neuron value that is figured on the neuron.

The output neurons with softmax activation shown on the right side use 100 pixels from left to right 0-100% to express the prediction accuracy.
If the prediction was the target the neuron becomes green and the class gold.

---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_spot_wrong.gif?raw=true)
*We can also spot what's going wrong. The state slider, correct = 0, incorrect = 1, all = 2. In combination with the other sliders every data is easily accessible. For a precise use. The sliders can be controlled by the left and right arrow keys on the keyboard. For example, to determine the start of training. 59916 seems a 7, or?*

Despite goodgame is kept really low with basic ideas and not a special neural network, the functionality can come really complex.
I was looking for a way to make it intuitiv to play. So with a left click you can put something in, with a right click you can take something out.

---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_user_samples.gif?raw=true)
*We can test the networks with our own samples.*

So we can also create, load and save our neural networks. 
Additional goodgame saves after a close and loads after a start your neural network automatically.
So you can do really strange things and compare the networks with specific examples. 

---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_compare_train.gif?raw=true)
*Another example of the functionality.*



---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_quantum_leaps.gif?raw=true)

*How the prediction moves from one class to the next has been one of the most interesting things to me. Further transformations, from '8' to '6' and from '6' to '5' can provide even more insights about the classes and their relationship.*

---

The quantum leaps of neural networks or just the change in classification of the nearby input increases the understand of neural networks a lot. Especially the intuition how the prediction was made and how we would evaluate this as a human. 

---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_dnn_basics_demo.gif?raw=true)
*The Class Accuracy on the left shows the prediction of each class. The question was, how many hidden neurons are needed to handle the data? The demo shows a test till a 784-7-10 network. The experiments can go further, but the 784-3-10 network suprised me most, but 784-4-10 network can handle all data in my opinion*

Logistic regression is like a neural network with one layer, the parameters to compute are here 784 * 10 = 7840 + 10 for the bias = 7850. A nerual network with one layer like the 784-7-10 computes 784 * 7 + 7 * 10 = 5558 parameters without a bias in the case of gg and can outperform logistic regression. Efficency is a core of gg, with one more layer, 784-7-100-10 the network would compute 6188 parameters, but how would we rate a 784-6-50-10 network with 5504 parameters? A very important aspect if we think about how we should build our networks, but also for the prediction quality.


---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_incorrect_custom_training.gif?raw=true)
*It is time for popcorn, take your seat and manipulate the predictions, train with your intuition within the training.


---
![alt text](https://github.com/grensen/gif_test/blob/master/Figures/gg_one_20_layers.gif?raw=true)
*How many layers can we train? This is a deep neural network with 20 layers. It was really hard to train, but the pattern of the neurons looks pretty cool.*

---

---
![alt text](https://github.com/grensen/gif_test/blob/master/Figures/gg_one_low_vs_high_lr.gif?raw=true)
*The learning rate affects the training. In case of the ReLU activation, the learning rate affects also the activation level of the neurons, lower lr's keep the activation level high, and high lr's keep the activations low, till a whole layer is disconnected. The example shows a briefly look into the test low = 0.001 vs high = 0.01 after 200.000 backpropagations, even with this moderate settings is this effect easy to see.*

---

---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_support_class.gif?raw=true)
*How to increase the weakest class prediction. If the step was wrong, take the last training step and try again. It looks not so good for the others classes after this move, but with a lot of sensitive it's possible to support your network with specific training.*


---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_smaller_classification.gif?raw=true)
*How would the the neural network perform with only three classes to predict? Experiments like this are not very useful, on the other they could bring new perspectives. I didn't expect this test accuracy after only one training, neat.*


---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_new_network_inside.gif?raw=true)
*With more neurons you reach more accuracy, that's right, almost. Neurons can be added or removed all the time with gg. Here the starting point was used to create a new network inside the existing one. After enough rounds the merged networks should be act as one, sometimes not so clever. It seems more useful to use the final size from start, but not always.*


---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_gg_outro.gif?raw=true)
*Finally, it is possible to add new data for new classes that differ from the common data. gg*






### How to install

[Download](https://drive.google.com/file/d/12s7E-2-GqgkYY6ZNw0jgKvGTeVDZbXqB/view) and extract the directory to the c: folder.

Or:
 1. Download the folder MNIST_Data for the unzipped data set and the Neural_Network_Backup with the empty file.
 2. Then create the directory c:/goodgame/one/ and put both folders inside.
 3. Now goodgame is ready to run on Visual Studio with the goodgame.cs code.
 4. To collapse all the 1400 lines press CTRL + M + O
 
 
---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_install_code.gif?raw=true)


Core functions:

---
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/core_functions.png?raw=true)
*The code is realy complicated in many parts, a good way to get the connection is start with the NeuralNetworkRun() function. 
The function handles training and test runs. NeuralNetworkSample() treats the custom training.*


---
Build a release version of your goodgame app.
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_build_releasel.gif?raw=true)
---
