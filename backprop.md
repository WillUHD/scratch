<div align="center">

# All About **Backpropagation**
###### **Will Chen** | SHSID Data Science Group

## Key Questions

<div align="left">
  
Backpropagation is an important step in supervised learning procedures. By the end of the lesson, you will be able to answer the following **key questions**: 
1.	What is backpropagation and how is it used? 
2.	What are the similarities and differences between forward propagation and backpropagation? 
3.	How does backpropagation specifically function? 
4.	Why is it such an important process and why is it used so much? 

<div align="center">
  
## Key Terms
<div align="left">
In order to understand this lesson, you would have a grasp of the following key concepts and terms: 

- **Layers**: Each network has many layers. The first layer takes in the input vector and the last layer is output. In the diagram below, each layer is one column. 

- **Vector**: A list of the numbers in each layer of a neural network. 

- **Neural Network**: _Input vectors_ (blue) are passed through the _hidden layers_ (green) to form an _output vector_ (yellow). 

- **Neuron**: A node in the network, each containing a special number called weight. In the diagram, it is each circle (🟢). When running the model, each neuron multiplies its weight with the previous layer’s vector and adds them together, forming its result and ready to be used by the next layer’s neurons.

- **Hidden Layer**: Contains a series of weights for each result of each previous layer to multiply with, producing its output value. 

- **Matrix Multiplication**: The method used to multiply vectors with weights. Each vector is multiplied by the weight once and passed on to the next layer. 

- **Epoch**: One training generation. One round of backpropagation (explained later) is used per epoch.

- **Loss**: During supervised training, the model’s output is continuously compared with the real outcome. The purpose of the model is to form results as close to the real outcome as possible, in other words, minimize the loss. The difference between output vector and the real outcome vector is known as loss.

- **Supervised training**: A type of training method where the expected input and result vectors are labeled for the model to train with. 


<img align="left" width="200" height="417" alt="image" src="https://github.com/user-attachments/assets/0a0752e0-6001-4c1a-a465-6ac34f0ac756" />

<img align="center" width="200" height="240" alt="image" src="https://github.com/user-attachments/assets/2f505d8a-e71f-408f-af05-3f597da55802" />

<img align="right" width="200" height="700" alt="image" src="https://github.com/user-attachments/assets/47bbc890-cec4-42b7-a689-c4fbacb51346" />

<div align="center">
  
## Introduction to Backpropagation
<div align="left">
  
### What is backpropagation?

**Backpropagation** is a crucial step in neural networks that improves the accuracy of a model by traversing a network in reverse to find out how much each weight contributed to the model’s inaccuracy. 

<img align="right" width="300" height="385" alt="image" src="https://github.com/user-attachments/assets/70fd512a-79ef-4dda-a297-626b5def20b5" />

As we know from **forward propagation**, multilayer perceptron networks work by moving through layers, from the input vector layer to the output. Back propagation, as the name suggests, moves in reverse: It **starts with the very last layer** (output) and **propagates back to the previous layers**. When forward propagation is useful in running the model, backpropagation is useful in training. 

**Think of it like solving a math formula**: You know you made a mistake along the way because your answer is different from the answer key. So, you want to look through your steps in reverse to adjust them. Similarly, when a model returns something we don’t expect, we can use backpropagation to **adjust each neuron until it returns the things we like**. 

### What happens in training?

Training a model is a** very repetitive process**. From earlier on, you might recall that **epochs** are the individual rounds of training, and that the goal of training a model is to **minimize the errors it makes**. During each epoch, several things happen: 

1.	We run the model through its neurons (aka Forward Propagation). 
2.	The model gives us its predictions (in the final output vector layer). 
3.	We compare the prediction with the real, expected result (given by our training dataset). 
4.	**We use backpropagation as a tool to know which neurons went wrong, and to what extent**. 
5.	We adjust the neurons and move on to the next epoch, and so on until we’re happy. 

<img align="right" width="300" height="456" alt="image" src="https://github.com/user-attachments/assets/2247773c-a586-487a-a8f3-e02cc36f4be0" />

**But what happens before any epoch**? We must set up the model. Usually, models begin with “blank” neurons, or neurons that don’t contain any special information about their weights and biases. When we run this blank model for the first time with our training features, it probably won’t return something we’d like. 

When we compare the model’s predictions with the labels, we get a difference, which is known as loss. This difference is crucial because it tells us if our predictions are close enough or not. But it doesn’t tell us what exactly went wrong. How do we know that? **The answer? We do it through backpropagation**. 

### How does backpropagation really work?

Think of it like **taking blame**. We start off with knowing a difference (or loss) between the expected and predicted results, which we don’t like. Of course, the neurons at the very end of the network are **most directly responsible for the loss** since they are closest to the output neurons, so they take all the blame first. Then, using a special math property called the **chain rule**, the blame is further **passed down along the line from back to front**, and **distributed across the entire network**. 

#### Understanding loss calculation

First, we need to get how the loss is exactly calculated. As we know, the model produces predictions that can be compared with expected predictions. For each forward propagation, loss is taken as a single number. Since there can be a lot of output neurons, we need to take the mean of the errors. 

But it’s not just any mean. Because we’re dealing with some calculus (later on), scientists realized that taking the **MSE (Mean Squared Error)** will make the calculations a lot easier than just taking the mean regularly, or MAE (Mean Absolute Error). **MSE is defined by the average squared difference between the predictions and expected results**. 

#### Dealing with gradients

A key optimizer within backpropagation is **gradient descent**. It’s a type of iterative optimization algorithm; in other words, it slowly solves a problem iteratively one by one. Its use in backpropagation is that it tells us **how much we need to change a parameter in order to minimize the loss function**.

<img align="right" width="300" height="373" alt="image" src="https://github.com/user-attachments/assets/5fcc7eb4-eb6d-4ad5-8858-50a32eb900f8" />

Reviewing some calculus terminology: 

-	The **gradient** of a function is a vector of its _partial derivatives_. 
-	A **partial derivative** is essentially the rate of change of an object in the x direction, shown by the black line on the right image. 
-	The **derivative** of the function measures the weight change with respect to the input’s change. It tells us the _direction of the function._ 
-	The **gradient** of the function, then, will tell us the magnitude of the function, otherwise _how much our parameters need to change_. 
-	**E_tot** is our _“surface”_ for gradient descent minimization, similar to a _“warped 3D surface”_ like the picture to the right. This picture shows 3 dimensions, assuming there are 2 weights to work with and 1 local gradient. 

With this knowledge in mind, our aims in backpropagation becomes clear. We want to: 

-	Get the _gradient_ of our error (magnitude to adjust our parameters)
-	The _opposite_ of our gradient would be the _direction for us to descend_ on E_tot. 
-	Taking the same image as above, gradient descent is similar to **rolling a ball down the hill**. 
-	When the ball reaches the very bottom, the slope of the surface is 0, and that’s the most optimal way we can adjust our loss to minimize it. 

In summary, the concept of Gradient Descent follows the derivatives to essentially “roll” down the slope until it finds its way to the minimum. 

#### Using the Chain Rule 

So, we know we need to find the gradient (the slope of our error surface) to roll the ball downhill towards the minimum loss. But a neural network has many layers and many weights. How do we figure out the specific gradient for a weight buried deep inside the network? This is where a fundamental concept from calculus, the **Chain Rule**, becomes the key to our lesson.

The Chain Rule allows us to calculate how one variable affects another indirectly, through a chain of intermediate variables. The final loss is directly affected by the output of the last hidden layer. The output of that last hidden layer is affected by its weights and the output of the layer before it, and so on. **The Chain Rule gives us a precise mathematical way to quantify and pass this blame backward through the network**.

For any given weight in the network, we want to calculate its "blame", or more formally, the partial derivative of the total error (E_tot) with respect to that weight (w). This tells us: "_If I change this specific weight just a tiny bit, how much will the total error of the network change_?"

Using the Chain Rule, we can break this down into **three manageable pieces** for a weight connected to an output neuron:

1.	How much did the total **error** change with respect to the neuron's **final output**? 
2.	How much did the neuron's **final output** change with respect to **its pre-activated input**? 
3.	How much did the neuron's **pre-activated input** change with respect to the **weight**? 

By multiplying these three rates of change together, **the Chain Rule gives us the overall gradient for that one weight**. For weights in earlier hidden layers, the chain just gets longer, but the principle is exactly the same. The error is propagated backward, layer by layer, **with each layer using the error signal calculated from the layer in front of it**.

#### Making adjustments with the Update Rule

Once backpropagation has used the Chain Rule to calculate the gradient for every single weight in the network, we know **two things** for each weight:

1.	The **direction** of the steepest increase in error _(the gradient itself)_.
2.	The **magnitude** of that slope _(how much the error will increase)_.

Since our goal is to decrease the error, we simply go in the opposite direction of the gradient. This brings us to the **Update Rule**. For each weight, _we perform the following calculation_:

$$ W_{new} = W_{old} - (LR * Gradient) $$

Let's break this down:

-	**Old Weight** (W_old): The current value of the weight before the update.
-	**Gradient**: The value we just calculated through backpropagation for this specific weight. It tells us which way is "uphill."
-	**Learning Rate**: This is a small number (e.g., 0.01) that we choose before training starts. It's a crucial parameter that controls _how big of a step we take downhill_. 
> If the learning rate is _too large_, we might overshoot the bottom of the valley and _end up on the other side_ (overfitting). 
> If it's _too small_, training will _take an incredibly long time and explode_, like taking tiny baby steps down a huge mountain (exploding). 

This update process is performed for _every weight and bias_ in the entire network during each training epoch.

#### Linear regression

If this still seems abstract, let's think about the simplest possible "network": a **Linear Regression model**. A linear regression model tries to fit a _straight line (y = mx + b) to a set of data points_.

In the context of this equation,

$$ y = mx + b $$

-	x is our input.
-	m (the slope) and b (the y-intercept) are our "weights" or parameters.
-	y is our prediction.

The "loss" is typically the **Mean Squared Error (MSE)** between our predicted y values and the actual y values from the data. To find the best line, we need to find the values of m and b that minimize this loss. How do we do that? With gradient descent! We would:

-	Calculate the partial derivative of the MSE **with respect to m**.
-	Calculate the partial derivative of the MSE **with respect to b**.
-	Use these gradients in the update rule to slowly _adjust m and b until the loss is at a minimum_.

Backpropagation is simply a **generalization of this exact process** for a much more complex model with many layers and non-linear activation functions. It's a clever, systematic way of applying gradient descent to millions of parameters simultaneously.

#### Other key aspects of backpropagation

##### Activation functions

Remember that each neuron applies an **activation function** (like Sigmoid or ReLU) to its input. When we use the Chain Rule, we need to calculate the _derivative of this activation function_. This is a critical reason why activation functions **must be differentiable** (i.e., we can find their slope at any point)._ A function with a "clean" and easy-to-calculate derivative makes the math of backpropagation much more efficient_.

##### Different versions of gradient descent

We don't have to calculate the loss over the entire dataset before making one single update. This would be very slow. Instead, we use different strategies:

-	**Batch Gradient Descent**: _The "classic" approach_. We run all training examples through the network, average their gradients, and then update the weights once. It's stable but _memory-intensive and slow for large datasets_.
-	**Stochastic Gradient Descent** (SGD): _The opposite extreme_. We update the weights after every single training example. It's much faster and can escape shallow local minima, but the _updates can be very "noisy" and erratic_.
-	**Mini-Batch Gradient Descent**: The _best of both worlds_ and the most common method. We divide the training data into small batches (e.g., 32, 64, or 128 examples), and we update the weights once per batch. This provides _a good balance between the stability of Batch GD and the speed of SGD_.

##### Vanishing and exploding gradients

Backpropagation isn't without its challenges. In very deep networks (networks with many layers), _the error signal can run into problems as it's propagated backward_.

-	**Vanishing Gradients**: As the gradient is passed back, it can be multiplied by numbers less than one over and over. This can cause the gradient to become incredibly small, effectively "vanishing" by the time it reaches the early layers. When this happens, the weights in the early layers stop updating, and the network stops learning.
-	**Exploding Gradients**: The opposite can also occur. The gradient can be repeatedly multiplied by numbers greater than one, causing it to become astronomically large. This leads to massive weight updates and makes the model unstable and unable to learn.

Modern deep learning has developed solutions to these problems, such as using specific activation functions like ReLU (whose derivative is a constant 1), better weight initialization methods, and normalization techniques, all designed to keep the backpropagated signal healthy and effective. 

<div align="center">

### Conclusion
<div align="left">

That’s all for our lesson on backpropagation! A lot of it relies on complex math, but it’s the heart of the model training process. This clever algorithm is what allows a neural network to truly learn from its mistakes by translating the final error into actionable feedback for every single weight.

By using the **chain rule** to propagate this error signal backward, it calculates the precise **gradient** needed to guide the model's optimization. These gradients ensure each **weight** is nudged in the exact direction that will **minimize the overall loss**. Repeated over thousands of **epochs**, this **iterative refinement** transforms a randomly initialized **network** into a powerful and accurate **predictive tool**. 

Ultimately, backpropagation is the engine that drives intelligence in most modern ML systems, turning abstract data into concrete knowledge. 

