# Playing Atari using Spiking Neural Networks


## Introduction

> Spiking neural networks are hard to train using gradient descent due to the
non-differentiable nature of the spike. Therefore, we train an artificial neural
network (ANN) with similar network architecture using the DQN algorithm
and transfer the learned weights to the SNN. We show that SNNs are capable
of outperforming ANNs with similar weights and network architecture.

This is the original line from the introduction part which suggests that they are not training SNNs directly, instead they are training ANNs and then they transferred the same weights to the simmilar SNN architechture and tried to show they work better than ANNs with simmilar weights and architecture.

---

## Methods
### Reward Structure
They modified the reward structure of the game. Generally we achieve higher rewards 

### Weights transfer
They trained an ANN to play Breakout using a DQN and then transfered the trained weights to a SNN with similar network architecture.

### Network architecture
To avoid complexity they removed convolution layers and directly used fully connected layers with 1 hidden layer.

ANN architecture:
```
Input(80*80) -> 1000 ReLU activated Neurons -> Output Layer of 4 neurons
```

SNN architecture:
```
Input(80*80) -> 1000 LIF Neurons(or with adaptive threshold) -> Output Layer with 4 LIF Neurons
```

### Hyper-Parameters
#### For DQN
| **Hyperparameter**              | **Value**      | **Description**   |
| ------------------------------- | -------------- | ----------------------|
| Training Length                 | 30000 episodes | Number of games the agent trains over.                |
| Mini-batch Size                 | 32             | Number of transitions sampled per gradient update.    |
| Replay Memory Size              | 200000         | Number of most recent transitions stored.             |
| Replay Memory Init Size         | 50000          | Initial transitions collected using random actions.   |
| Agent History Length            | 4              | Number of most recent frames given as input.          |
| Target Network Update Frequency | 10000          | Number of updates before syncing with target network. |
| Discount Factor (γ)             | 0.99           | Used for estimating optimal action-values.            |
| Frame Skip                      | 3              | Number of frames skipped between each state.          |
| Action Repeat                   | 4              | Number of times each selected action is repeated.     |
| Update Frequency                | 4              | Steps between each gradient update.                   |
| Update Rule                     | RMSProp        | Optimization algorithm used.                          |
| Learning Rate                   | 0.00025        | Learning rate for RMSProp.                            |
| Gradient Momentum               | 0.95           | Momentum for gradient in RMSProp.                     |
| Squared Gradient Momentum       | 0.95           | Momentum for squared gradient in RMSProp.             |
| Min Squared Gradient            | 0.01           | Small constant to avoid division by zero in RMSProp.  |
| Initial Exploration (ε)         | 1.0            | Initial ε for ε-greedy exploration.                   |
| Final Exploration (ε)           | 0.1            | Final ε for ε-greedy exploration.                     |
| Final Exploration Step          | 200000         | Steps over which ε decays to final value.             |
| Reward                          | 1.0            | All rewards scaled to +1 (no negative rewards).       |

#### For SNN

| **Hyperparameter** | **Value** | **Description** |
| ------------------ | --------- | ---------------------------------------|
| Refractory Period  | 0         | Time (in ms) during which the neuron cannot spike after a spike. |
| Threshold Voltage  | -52       | Membrane potential at which the neuron spikes.                   |
| Resting Voltage    | -65       | Membrane potential when the neuron is at rest.|
| Voltage Decay      | 0.01      | How much voltage decays at each time step.|
|Time steps | 500 | Time steps for accumulating the voltage |

#### SNN with Adaptive Threshold

| **Hyperparameter** | **Value** | **Description** |
| ------------------ | --------- | ---------------------------------------|
| Theta plus | 0.05 | Amount of threshold increased after each spike. |
| Theta decay | 1x10−7 | Time constant of adaptive threshold decay. |

## Experiments and Results

Two different Input methods which are Binary and Gray-Scale were tried and for both of them Simple SNNs v/s stochastic SNNs(There is a complete section over this concept) were used. Additionally Time-to-first-spike coding was also explored.

### Binary Input
#### Pre-Processing

```
Raw Game Frame (e.g., Breakout)
      ↓
Resize / Crop to 80×80
      ↓
Convert to Grayscale
      ↓
Binarize the Image (0s and 1s)
      ↓
Subtract Previous Frame from Current
      ↓
Make Negetive Values 0
      ↓
Add 4 most recent Frames

(Although i feel that some pixels may add up to make the value more than 1)
(But in the paper it is not mentioned to re-binarize again after adding)

```
which resulted into something like:


<img src="https://hackmd.io/_uploads/H1wrXZNSlx.png" alt="Alt Text" width="50%" />

Notice that we can see the ball nd paddle moving but we don't know the direction of motion

#### ANN Performance
Consistant score of 6 tested with no exploration.

#### SNN Performance
- Mean: 5.25 
- STD: 2.14

The first layer of weights are scaled by 10x and the second layer of weights are scaled by 100x. The hidden layer uses LIF neurons with adaptive threshold.

#### Time-to-first-spike coding
- Mean: 5.08
- STD: 2.00

The network has similar performance but is 3 times faster than rate-based
coding.

#### Stochastic SNNs
- Mean: 7.85
- STD: 1.87

The first layer of weights are scaled by 10x and the second layer of weights are scaled by 100x. The hidden layer uses LIF neurons with adaptive threshold.

#### Stochastic SNNs with Exponential Escape Noise Function
- Mean: 7.48
- STD: 2.37

The first layer of weights are scaled by 10x and the second layer of weights are scaled by 100x. The hidden layer uses LIF neurons with adaptive threshold.

### Grey-Scale Input

#### Pre-Processing
Vanilla greyscale input lacked the information about direction of ball and paddle. Hence to solve this problem, each of the 4 frames were weighted according to time and stiched together. At time $t$, the state is made up of the sum of the most recent 4 frames as follows:
\begin{equation}
S_t = F_t \cdot 1 + F_{t-1} \cdot 0.75 + F_{t-2} \cdot 0.5 + F_{t-3} \cdot 0.25
\end{equation}
where $S_t$ and $F_t$ are the state and the frame at time $t$, respectively. This resulted into input of the following kind which caputed the direction of motion:

<img src="https://hackmd.io/_uploads/HkqWNW4Hle.png" alt="Alt Text" width="50%" />


#### ANN Performance
- Mean: 9.32
- STD: 0.63

> Now for SNN encoding the authour tried 2 approaches one was
> - At each time step, the input neuron spikes with the probability
equal to the value of its corresponding pixel
> - uses the actual grayscale image as the input. Therefore, instead of a binary spike, we value of the spike is equal to the intensity
of each pixel.

#### SNN Performance with Spike Encoding
No results share but author tells that: "The first method of input does not maintain the relationship between the intensity values of the image and therefore the network does not receive the weighted time data. Due to this reason, the first method does not perform very well."

#### SNN Performance with Greyscaled Images
- Mean: 10.05
- STD: 0.68

The first layer of weights are scaled by 5x and the second layer of weights are the same. The neurons used for this experiment are LIF neurons.


#### Stochastic SNNs with Greyscale inputs
- Mean: 5.37
- STD: 1.52

The first layer of weights are scaled by 180x and the second layer of weights are also scaled by 180x. The neurons in the second layer have adaptive threshold.

No Explaination given for such a poor performance of Stochastic SNNs.

### Robustness
To test the robustness of the SNN against the ANN, author tested the performance of each network when the three horizontal bars of pixels of the input are occluded. The performance is tested for every possible position of the bar on the screen.



<img src="https://hackmd.io/_uploads/B12d5W4Hex.png" alt="Alt Text" width="70%" />


SNNs are much more robust compared to ANN even though they share the same weights. The artificial neural network trained using the DQN algorithm is sensitive to changes at a few places in the input. When these areas are occluded, the ANN performs poorly. Surprisingly, occluding these areas does not affect the performance of the SNN.

---







<!-- ### Preprocessing of data:
Each frame from the gym environment was cropped to remove the text above
the screen displaying the score and the number of lives left. The image was then re-sized to a 80x80 image and converted to a binary image. The previous frame was then subtracted from the current frame while clamping all the negative value to 0. Then the most recent 4 difference frames were added to create a state. 
Thus, a state is a 80x80 binary image containing the movement information of the last 4 states.

<img src="img/image.png" style="display: block; margin-left: auto; margin-right: auto;" width="300"/>

<p align="center"><strong>Your caption or description</strong></p>

<div align="center">

<img src="" width="300"/>
**The image above shows the movement of the ball and the paddle, however, it is not
possible to detect the direction of the movement from the image.**

</div>

---

![](https://hackmd.io/_uploads/BkHhqgVSge.png)
> This Image shows the ANN performance using DQN algorithm only tested for 100 Games of Atari Breakout. -->


<!-- ## Approaches for SNN:

### The choice dilemma:
+ There is an ongoing debate on two methods for encoding the spikes. Namely, rate-based coding versus spike-based coding.
+ However, the author came to a result that performance of the spike-based coding is 3 times faster than rate-based coding.

### Scaling of weights:

+ Due to membrane leaking, the voltage with every successive layer, we feel the need that we should scale the weight for every input layer successively, instead of just copying it directly from SNN. This makes sense because as we go deep into our network layers, the spikes gets more and more sparse.

+ Hence, the author scaled the first layer of weights by 10x and second layer by 100x. -->

## Conclusion

| Method of input | ANN | SNN | Stochastic SNN| 
|-------|-------- |------ |------|
Binary | 6.0 ± 0 | 5.25 ± 2.14 | 7.58 ± 1.88
Grayscale | 9.32 ± 0.6 | 10.05 ± 0.6 | 5.37 ± 1.52

1. Spiking neural networks can be used to represent policies for reinforcement learning tasks like playing Atari games.
2. Spiking neural networks can be trained by transfer of weights from artificial neural networks.
3. Spiking neural networks can outperform the artificial neural network from which the weights have been transferred on reinforcement learning tasks like playing Atari games.
4. Spiking neural networks are more robust to attacks and perturbations in the input. They are also more generalized and perform better on states that they have not encountered before.




## Theory for Some Concepts

### Concept of Stochastic LIF neurons:

+ After transferring the weights from ANN to SNN, performance of the model dipped. This was accounted to the reason that REctified linear units (RElu) was used as an activation function in ANN. It gave linear output for positive values and zero for negative values.

+ Spiking Neurons, however, communicated via spikes which were binary in nature. Hence, Spiking neurons never gave any outputs if their membrane potential was below threshold.

+ To fix this problem, stochastic LIF neurons were added which had a chance to spike at each step with corresponding probability proportional to how close 


<img src="https://hackmd.io/_uploads/rJ4Qwg4Sgx.png" alt="Alt Text" width="70%" />
>This Curve shows the probability for stochastic spike is 1 at threshold and 0 at Reset voltage.

+ The probability to spike below threshold potential is also called **Escape Noise** in SNN models. Here, instead of being a linear function, the escape noise $\sigma$ was a bounded exponential function.

\begin{equation}
\sigma(V) = 
\begin{cases}
\frac{\delta t}{\tau_\sigma} \exp\left( \beta_\sigma (V - \theta) \right), & \text{if less than 1} \\
1, & \text{otherwise}
\end{cases}
\end{equation}

> Here V is membrane potential, $\theta$ is threshold potential and $\delta$t is duration of time step.
> $\tau_\sigma$ and $\beta_\sigma$ constant positive parameters. (Both are taken to be 1 to produce the curve shown above.)



+ This boosted the performance significantly and network performed better on average than SNN.

<!-- ### Grayscale Input problem:
+ As mentioned above, vanilla greyscale input lacked the information about direction of ball and paddle. Hence to solve this problem, each of the 4 frames were weighted according to time and stiched together.

+ At time $t$, the state is made up of the sum of the most recent 4 frames as follows:
\begin{equation}
S_t = F_t \cdot 1 + F_{t-1} \cdot 0.75 + F_{t-2} \cdot 0.5 + F_{t-3} \cdot 0.25
\end{equation}
where $S_t$ and $F_t$ are the state and the frame at time $t$, respectively.

![](https://hackmd.io/_uploads/ByeoYe4Hxg.png)
> We see how this kind of input helps our model to recollect the sense of direction for ball and paddle. -->

### Concept of Time-to-first-spike coding
Instead of how often a neuron fires , we care about how soon it fires after a stimulus is applied. So a neuron will keep accumulating voltage and fire only once and what matters is how soon it fired.


## ANN to SNN conversion Related works mentioned:
- [Mapping from frame-driven to frame-free event-driven vision systems by low-rate rate coding and coincidence processing–application to feedforward convnets.](https://www.researchgate.net/publication/256837357_Mapping_from_Frame-Driven_to_Frame-Free_Event-Driven_Vision_Systems_by_Low-Rate_Rate-Coding_and_Coincidence_Processing_Application_to_Feed_Forward_ConvNets) \[PerzeCarrasco et al. (2013)\]

(Basically they first introduced the idea of converting CNN to spiking
neurons with the aim of processing inputs from event-based sensors)

- [Spiking deep convolutional neural networks for energy-efficient object recognition.](https://www.researchgate.net/publication/276481463_Spiking_Deep_Convolutional_Neural_Networks_for_Energy-Efficient_Object_Recognition) \[Cao et al. (2015)\]

(suggested that frequency of spikes of the spiking neuron is closely
related to the activations of a rectified linear unit (ReLU) and reported good performance on computer vision benchmarks)

- [Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing. 2015 International Joint.](https://ieeexplore.ieee.org/document/7280696) \[ Diehl et al. (2015)\]

(proposed a method of weight normalization that rescales the weights of the SNN to reduce the errors due to excessive or too little firining of neurons. They also showed near loss-less conversion of ANNs for the MNIST classification task.)

- [Evaluation of event-based algorithms for optical flow with ground-truth from inertial measurement sensor.](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2016.00176/full) \[Rueckauer et al. 2016\]

- [Theory and tools for the conversion of analog to spiking convolutional
neural networks](https://arxiv.org/abs/1612.04052) \[Rueckauer et al. 2016\]

(The Above two demonstrated spiking equivalents of a variety
of common operations used in deep convolutional networks like max-pooling,
SoftMax, batch-normalization and inception modules. This allowed them to
convert popular CNN architectures like VGG-16, Inception-V3, BinaryNet, etc.
They achieved near loss-less conversion of these networks.)
