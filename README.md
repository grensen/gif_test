# goodgame|one Test Release
## Play Neural Networks to the End

goodgame or shorter gg is a huge trade-off to express neural networks in a new way and work with them.
gg comes from e-sports and was the name of my team years ago, but more important, after a game the teams say gg for a good time, 
to make the long story short, gg treats neural networks like a good game.

Let me give a first example:
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_hello_goodgame.gif?raw=true)



If you watched the neural network series from ![3blue1brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), you know the 784-16-16-10 neural network already.
That's important because the starting point with goodgame is that network with the default hyperparameters.
After a first MNIST training over 60000 samples the resulting test accuracy brings 93.29%, that is pretty good the best I know.

Let's create a sample and train it:
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_create_inputs_and_train.gif?raw=true)


Before we go further let's checkout what goodgame can show, the input neurons are red rectangles on the left and express the MNIST samples.
Every input neuron is fully connected with a weight to every output neuron on the next layer.
The weights in green chartreuse show positive values and the reds show negative values.
The highest or lowest color depends on the highest or lowest weight here and all other weights show colors between.
Note, the values are the colors in a good, but not an exact way. 
If the highest weight value will be 1.01 on that layer, the range can be really big and values with 0.07 and 0.04 can show the same color.

The neurons show their connections only if they are ReLU activated, if the value was below 0 the neuron is set to 0.
Here we can get a good feeling how the weights works, the highest neuron color represents also the highest neuron value that is figured on the neuron.

The output neurons with softmax activation shown on the right side use 100 pixels from left to right 0-100% to express the prediction accuracy.
If the prediction was the target the neuron becomes green and the class gold.

We can also spot what's going wrong:
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_spot_wrong.gif?raw=true)


Despite goodgame is kept really low with basic ideas and not a special neural network, the functionality can come really complex.
I was looking for a way to make it intuitiv to play. So with a left click you can put something in, with a right click you can take something out.


We can test the network with our own samples:
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_user_samples.gif?raw=true)

So we can also create, load and save our neural networks. 
Additional goodgame saves after a close and loads after a start your neural network automatically.
So you can do really strange things and compare the networks.

Another example of the functionality:
![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_compare_train.gif?raw=true)


~~To install goodgame, download the files and put them into the c:/goodgame/one/ directory and start the game.~~
