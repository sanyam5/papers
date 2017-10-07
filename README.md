# papers
List of papers I have read, am reading and want to read starting 1st Sept 2017.

## Read

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf) - ICML 2016 - [RL]

  <details>
  aka A3C. Instead of training on samples from replay memory to decorrelate temporal relations, use multiple agents operating in their own copy the environment using a current global policy. Training becomes more stable. Beats the previous best in half the training time. Train k agents on a single k-core CPU. No communication costs as with [Gorrila](https://arxiv.org/abs/1507.04296). In case of off-policy learning the individual agents can apply different policies which is more explorative and stable. Replay memory can still be used with this to increase data-efficiency.
  </details>

- [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495) - 2014 - [CV] [GANs]
  <details>
  It's a GAN in disguise. You datasets from 2 domains - 1) labelled synthetic image classes and 2) unlabelled real images. You want to label the images from real domain. Idea: There are three NN modules - G, C and D. Domain invariant features must be learn by network G. Feed the features to their equivalent of a discriminator (D), penalize N if D can predict domain from given features. Also feed the same features to classifier C train it to label the synthetic data. Over time D can't tell the domain, the features learnt are domain-invariant and by the covariate shift assumption network [G --> C] becomes good at classifying unlabelled real images.
  </details>

- [Learning to Repeat: Fine Grained Action Repetition for deep reinforcement learning](https://arxiv.org/pdf/1702.06054.pdf) - ICLR 2017 - [RL]
  <details>
  aka FiGAR. In policy gradient method, instead of just predicting the next action `a` from a set of actions `A` (continuous or discrete) predict a tuple (`a`, `w`) from `A` (actions) and a set of discrete integers `W`. Repeat action `a` for the next `w` time-steps. The intuition is this: in many situations you want to repeat the same action over a long range of time-steps. Decouple the prediction of `a` from `w` prevent the network from blowing up.
  </details>

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) by Ashish Vaswani et. al - 2017 - [DL]
  <details>
  <img src="attention-all.JPG"> <img src="attention-all-multi.JPG"> <p> novelty - 10/10. Fixed number of Attend and Analyse steps == number of stacked Transformer units (6 in the paper). Transformer unit: Consists of 1) an encoder layer 2) a decoder layer. Both layers contain a sub layer for attention and a fully connected sub-layer. The decoder contains and addition masking layer for preventing the decoder from seeing current and future token. Multiple smaller attention heads used instead of single big attention head. Positional information of both input and output sequences are fused into the embeddings before feeding it to the first Transformer layer. After that the order input or output tokens doesn't matter until the next Transformer unit. Positional encoding is cleverly designed to support relative indexing for attention. </p>
  
  </details>
  
- [Residual Algorithms: Reinforcement Learning with Function Approximation](http://www.leemon.com/papers/1995b.pdf) by L. Baird - 1995 - [RL]
  <details>
  TD(0) updates guaranteed to converge for table lookup but not for function approximators. Enter, Residual Gradient updates: Define a loss function E over the Bellman residue (RHS-LHS of Bellman eq.). Do gradient descent on w.r.t to E --> Guaranteed to converge but slow. Slow because the updates go both ways (next_state_action <--> this_state_action). Enter, Residual (delta_w_r) updates: Hit a compromise b/w TD(0) (delta_w_d) and Residual Gradient (delta_w_rg).
  
  TD(0) update
  <img src="td0.JPG">
  
  Residual Gradient update
  <img src="resgrad.JPG">
  
  <img src="residual.JPG">
  
  Dotted line is the hyperplane perpendicular to the true gradient w.r.t residue (need to stay left of it for robustness). Mustn't go far from TD(0) update (the direction of fast learning). Idea: take projection of TD(0) update w.r.t dotted line, nudge it slightly to the left.
  
  </details>

- [Efficient per-example gradient computations](https://arxiv.org/pdf/1510.01799.pdf) by Goodfellow - 2015 - [DL]
  <details>
  How to calculate norm of the gradient of each example in a batch? Naive: have N batches of size 1. Better approach to calculate the gradient of loss (which is the sum of errors on all examples in the batch) w.r.t all intermediate activations of all examples in the batch Z. And use this gradient Z-bar to compute norm of per-layer per-example.
  </details>

- [Differential training of Rollout policies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.2646&rep=rep1&type=pdf) by Bertsekas - 1997 - [RL]
  <details>
  Instead of approximating Q(s,a) or V(s) which are prone to noise in the environment and training (two-way flow of information),  approximate G(s,s') = V(s) - V(s') which tells how good is state s w.r.t. to s'. Interestingly standard RL methods can still be applied to approximate G. The states for this problem are (s,s') pairs and the reward is (r - r').
  </details>

- [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828.pdf) - CVPR 2017 - [GANs]
  <details>
  CVPR best paper award. Need for more annotated training data. The idea is to generate realistic images with class annotations from computer generated simulations. The generator G takes as input a computer generated simulation with a class label(like apple or orange) and makes changes to it so that it looks realistic. The dicriminator D must learn to discriminate the real images from the seemingly real ones generated by G. What if G takes a simulated image of an orange and changes it so much that it now looks like an apple?? We can't let this happen otherwise we will need somebody to re-annotate the generated images (which beats the whole purpose of automatically generating the annotated data). To prevent this, both G and D are allowed to focus on small regions of the image. This way G will never be able to make strong global changes. So class labels are preserved.
  </details>
  
- [Understanding the difficulty of training deep feedforward neural networks](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf) by Glorot and Bengio - 2010 - [DL]
  <details>
  Pre-batch normalization era: How factors such as initialization and non-linearities affect the training using SGD. Good initialization as shown by unsupervised pre-training (training each layer and its transpose to be an autoencoder) plays an important role in quick training. The activation functions should be zero-mean. The best non-linearity is cousin of tanh --> softsign (x/(1+|x|)). The best initializations have zero-mean and unit-variance. 
  </details>

- [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) by Mnih et. al - Nature 2015 - [RL]
  <details>
  Two extremely simple ideas. 1) Use experience replay - The order in which you provide observations (s, a, r, s`)  matters. If you provide them as they come it makes Q-learning unstable for function approximators because of the correlations b/w subsequent observations. Store observations in a buffer and provide them at random. 2) Use two (instead of one) Q networks. Freeze one and use as it base for evaluating the next state value for improving the second one. After C steps change the weights of the frozen network to be exactly same as the improved network and freeze it again.. loop.
  </details>


## Reading

- [A Distributional perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf) by Bellemare et. al - 2017 - [RL]


## Want to Read

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - NIPS 2013 - [RL]
- [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/pdf/1708.05866.pdf) - IEEE 2017 - [RL] [Survey]
- [Skip-Thought Vectors](http://papers.nips.cc/paper/5950-skip-thought-vectors.pdf) - NIPS 2015 - [NLP]
- [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf) - 2017 - [GANs]
- [Strategic Attentive Writer for Learning Macro-Actions](https://arxiv.org/pdf/1606.04695.pdf) - NIPS 2016 - [RL]
- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf) - 2017 - [RL]
- [Deep Photo Style Transfer](https://arxiv.org/pdf/1703.07511v1.pdf) - 2017 - [DL] [CV]
- [Adam: A method for stochastic optimization](http://arxiv.org/pdf/1412.6980) - 2014 - [Optim]
- [Semantic understanding of scenes through the ADE20K dataset](https://arxiv.org/pdf/1608.05442) - 2016 - [CV]
- [Representation Learning: A Review and New Perspectives](https://arxiv.org/pdf/1206.5538.pdf) - 2013 - [DL] [Survey]
- [Show, attend and tell: Neural image caption generation with visual attention](https://arxiv.org/pdf/1502.03044.pdf) - [CV]
- [A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](https://arxiv.org/pdf/1605.06069.pdf) - 2016 - [NLP]
- [Hierarchical multiscale recurrent neural networks](https://arxiv.org/pdf/1609.01704) - 2016 - [RNNs]
- [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) - 2014 - [GANs]
- [An actor-critic algorithm for sequence prediction](https://arxiv.org/pdf/1607.07086) - 2016 - [RL]
- [Unitary evolution recurrent neural networks](http://www.jmlr.org/proceedings/papers/v48/arjovsky16.pdf) - ICML 2016 - [RNNs]
- [Deconstructing the ladder network architecture](http://www.jmlr.org/proceedings/papers/v48/pezeshki16.html) - ICML 2016 - [DL]
