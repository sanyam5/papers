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


- [A Distributional perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf) by Bellemare et. al - 2017 - [RL]
  <details>
  Instead of modelling the expected reward, model a distribution over possible reward values. Stabilises training and capable of modelling intrinsic stochasticity in the environment and in the behaviour of the agent. Define equivalents of Bellman Operator and Bellman Optimality Operators in the distributional sense. They prove the Evaluation setting to be a contraction w.r.t to a particular metric - Wasserstein metrci. The Control setting however is not a contraction in any known metric. But it remains to be seen whether this presents a practical problem or not. 
  </details>

- [VAE: Auto-encoding variational bayes](https://arxiv.org/pdf/1312.6114) by Kingma et. al - 2014 - [Bayesian] [Unsupervised]
  <details>
  Understood it through this [Tutorial](https://arxiv.org/pdf/1606.05908.pdf) and  this [blog](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder). I am yet to fully grasp this from a theoretical side but from a deep learning side I think I understood this. This paper's main contribution to the AutoEncoder framework in my opinion was the fact that they perturbed the latent embeddings and made sure that the Decoder was still able to reconstruct it. But the main flaw is that the loss they use is between pixel to pixel (or dimension to dimenstion) with a complete disregard to the inter-pixel or inter-dimensional dependencies. I think this the primary reason why the generated and reconstructed images are fuzzy. Other recent papers like PixelVAE solve this problem in the image domain by using 5x5  pixelCNN autoregeressive decoder.
  
- [PixelVAE: A Latent Variable Model for Natural Images](https://arxiv.org/pdf/1611.05013) - By Gulrajani et. al - 2017 - [DL] [VAE]
  <details>
  The major contribution on the VAE architecture is that they use teacher forcing in decoder while training using PixelCNN. This frees the latent embedding from having to memorise fine details in images. How do they guarantee that semantic information flows throught the latent space while only the style information is flows through the PixelCNN? They use a 5x5 kernel from which it is impossible to get the big picture (pun, got it?). They are able to generate sharp images through it.

- [Skip-Thought Vectors](http://papers.nips.cc/paper/5950-skip-thought-vectors.pdf) - NIPS 2015 - [NLP]
  <details>
  Aim to construct semantic embeddings for sentences. Idea: given a sentence in a running text try to predict the previous sentence and the next sentence. Teacher force while predicting. If domain contain huge number of unique words, map them to the latent space of word2vec and then take the nearest neighbour in the small set of words that we want to consider. Test on downstream tasks, may put just one linear layer for adapting sentence embeddings to the task.

- [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf) - By C Zhang et al - 2016 - [DL]
  <details> Shows that a sufficiently large (with just 2*n+d parameters) network can overfit on a completely random dataset of n d-dimensional points. This shows that Neural Networks generalize well beyond the training dataset even though they have the power to overfit. Overfitting does require more time converge though. Maybe the reason the NNs generalize so well is that reaching generalizing solutions is somehow easier. </details>

- [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf) - By S Sabour et al - 2017 - [DL] [CV]
  <details>
  Building blocks of a NN are vectorized capsules as opposed to scalar neurons. Network formed by layers of capsules. The output of each capsule is a squished vector with a max lenght of 1. Each capsule (a capsule for detecting a nose for example) in the lower layer distributes its output to all capsules (a capsule for detecting the face) in the next layer. The distributed outputs are weighted according to a routing matrix C. The distributed outputs undergo an affine transformation (how is the existence and pose of nose related to the existence and pose of the face) by W matrix of the higher layer. These affine transformations from each of the lower capsules to a higher capsule are then summed together to form the resultant vector for the higher level capsule. The routing matrix C is calculated by the agreement between the affine transformations from the lower layer and the resultant vector. But this is a chicken and egg problem since we don't have the resultant vector without C. Therefore, The matrix C is iteratively (iter=3) calculated from scratch in every forward pass using the agreement (dot-product) b/w the supplied output from a particular lower level capsule and the resultant vector.
  
  I found the idea pretty interesting but I wish there was a more elegant way of calculating the routing matrix. The ad-hoc way of calculating the routing matrix leaves the possibility of instability in training a likely possibility.
  </details>
  
- [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf) - By A Oord et al - 2017 - [UL]
  <details>
  How do you train an auto encoder with an autoregressive decoder. How do you ensure that the latent representations learn a global aspect of the input and not some style characteristic of the input. After all, you are just minimising the MSE reconstruction loss. The model is free to choose what information it channels throught the latent representation and what information it channels through the autoregressive mechanism. One solution to this problem is making the latent space K-way categorical for a small and finite K like K=512.
  VQ-VAE: Just like an ordinary VAE except that the latent space Z has some K special vectors e1, e2, e3...eK. Encoder computes a continuous z. The special vector e_i nearest to z is passed on to Decoder. e_i is artificially given the gradients of z. But how are these special vectors selected? The special vectors are randomly initialised and then updated at every iteration to minimize the l2 loss between any given z and the special vectors. The special vectors play catch-up. What if the z vectors rush outwards too fast for the special vectors to catch-up. Don't worry we got an l2 loss for that too.   
  </details>

- [GENERALIZING ACROSS DOMAINS VIA CROSS-GRADIENT TRAINING](https://openreview.net/pdf?id=r1Dx7fbCW) - By Shankar et al. - ICLR 2018 - [Domain Adaptation] [UL]
  <details>
  [Do not understand some parts, will come back to it later] Awesome paper in my opinion. Assume you have a lot of training data in one domain and a little data for few other domains. How do you train a Neural Net which generalizes to data from a huge number of unseen domains? How can we leverage sparse data from few domains using a lot of data in one domain? Train two neural networks. First, standard, given a sample predicts the class label. Second NN helps in augmenting the data from sparse domains. How? Second NN is trained to predict the **domain** of the input. Augmentation is performed by perturbing the input so as to increase the loss of the second NN. Use the augmented input for training the first network. Interestingly, since the perturbations happen on a real space, the augemented input might not even belong to any of the few domains. It could be mutant domain of the domains under consideration. There is one subtle challenge though that the reader is quite likely to skim over--the perturbations must be such that they only disturb the domain of the input and not its label. I do not fully understand this yet. Will get back to this when I have more time.
  </details>

## Reading

- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by LeCun et. al - 1998 - [DL]
- [Human-level concept learning through probabilistic program induction](http://cims.nyu.edu/~brenden/LakeEtAl2015Science.pdf) by Lake et. al - 2015 - [Bayesian]


## Want to Read
- [Emergence of Invariance and Disentanglement in Deep Representations](https://arxiv.org/pdf/1706.01350.pdf) - 2017 - DL
- [DRAW: A recurrent neural network for image generation](https://arxiv.org/pdf/1502.04623.pdf) - 2015 - DL
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - NIPS 2013 - [RL]
- [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/pdf/1708.05866.pdf) - IEEE 2017 - [RL] [Survey]
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
- [Backpropagation through the void: Black-box gradient estimation](https://arxiv.org/pdf/1711.00123.pdf) - 2017 - [DL]
- [LONG SHORT-TERM MEMORY](http://www.bioinf.jku.at/publications/older/2604.pdf) - 1997 - [DL]
- [Contractive Auto-Encoders: Explicit Invariance During Feature Extraction](http://www.icml-2011.org/papers/455_icmlpaper.pdf) - ICML 2011 - [Unsupervised]
- [Semi-supervised Learning with Deep Generative Models](https://arxiv.org/pdf/1406.5298.pdf) - 2014 - [Semi-supervised]
- [Energy based Generative Adversarial Networks](https://arxiv.org/pdf/1609.03126.pdf) - 2017 - [GANs]
