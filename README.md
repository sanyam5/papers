# papers
List of papers I have read, am reading and want to read starting 1st Sept 2017.

## Read
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf) - ICML 2016 - [RL]
  - aka A3C. Instead of training on samples from replay memory to decorrelate temporal relations, use multiple agents operating in their own copy the environment using a current global policy. Training becomes more stable. Beats the previous best in half the training time. Train k agents on a single k-core CPU. No communication costs as with [Gorrila](https://arxiv.org/abs/1507.04296). In case of off-policy learning the individual agents can apply different policies which is more explorative and stable. Replay memory can still be used with this to increase data-efficiency.

- [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495) - 2014
  - It's a GAN in disguise. You datasets from 2 domains - 1) labelled synthetic image classes and 2) unlabelled real images. You want to label the images from real domain. Idea: There are three NN modules - G, C and D. Domain invariant features must be learn by network G. Feed the features to their equivalent of a discriminator (D), penalize N if D can predict domain from given features. Also feed the same features to classifier C train it to label the synthetic data. Over time D can't tell the domain, the features learnt are domain-invariant and by the covariate shift assumption network [G --> C] becomes good at classifying unlabelled real images.
  
## Reading
- [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828.pdf) - CVPR 2017 - [GANs]
  - Generating original-like synthetic data using GANs

## Want to Read
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - NIPS 2013 - [RL]
- [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/pdf/1708.05866.pdf) - IEEE 2017 - [RL] [Survey]
- [Learning to Repeat: Fine Grained Action Repetition for deep reinforcement learning](https://arxiv.org/pdf/1702.06054.pdf) - ICLR 2017 - [RL]
- [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf) - 2017 - [GANs]
