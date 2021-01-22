Final project for EECS 545: Machine Learning, by Jack Weitze and Hyeonsu Do (https://github.com/hdo0947).

This project was an investigation of iterative pruning in neural networks with weight rewinding. This methodology was introduced by Frankle and Carbin, 2018 (https://arxiv.org/abs/1803.03635).
The goal of pruning in neural networks is to set certain weights to 0, effectively reducing the number of trainable parameters, while maintaining model performance.

Choosing weights with the lowest magnitude is an effective method for pruning, as low-magnitude weights will tend to have low influence on the output. To achieve a desired level of sparsity,
one-shot pruning methods remove the corresponding lowest-magnitude weights in a single step. In contrast, in iterative pruning a portion of weights are pruned at regular intervals during training, and
the process repeated until the desired sparsity is achieved. Iterative pruning can achieve higher levels of sparisty for comparable final model performance, as compared to one-shot pruning methods.

Weight rewinding can allow for even sparser models, and empirically can sometimes improve final model performance. After each iterative pruning step, the weights are rewound to their initial values before
any training, then the training process restarted. The Lottery Ticket Hypothesis generally states that current neural network architectures contain these sparse, trainable subnetworks, which can be found via
iterative pruning with weight rewinding, and which achieve final performance at least at the level of the unpruned model. 

It's perhaps important to note that these are all unstructured pruning methods, which means that there won't be any immediate computation or memory efficiency improvements due to pruning unless sparse-matrix operations are used. Structured pruning methods, which essentially remove rows/columns from linear layer matrices or channels from convolutional filters, can improve computation and memory efficiency with no changes in operations, but these methods tend to be less effective for pruning (that is, an unstructured pruning method tends to achieve a higher level of sparsity than a structured method, for a comparable final model performance).

For our project, we were able to prune roughly 80% of ResNet18 weights over 7 pruning iterations on the CIFAR10 dataset. The final pruned model improved test accuracy by about 2% over the unpruned model
after a single iteration of training. We also prune a graph convolutional network (GCN) on the Cora dataset, where we were able to prune 98% of weights before accuracy began to decline.

Finally, we investigated how reusing a model from an earlier pruning iteration could change the pruning process. Specifically, we experimented with using knowledge distillation, with the trained, unpruned model 
as a teacher network to generate soft-labels for the pruned model currently being trained (the student network). We hypothesized that the soft-labels could improve the training of the pruning process. In a preliminary
experiment, we found that incorporating knowledge distillation can improve model performance. However, this experiment only considered the first portion of the training procedure on the first pruning iteration, so more comprehensive tests are needed to draw meaningful conclusions about this method.

More details on our work can be found in the final project report.