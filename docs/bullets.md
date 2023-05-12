# Bullet pints for writeup

## NN

### Design

* **Activation function** - Gelu
  * allows gradients to propagate allowing more nuance.
  * Add or take out normalisation.
  * Reference:
    * Hand writting dataset (MNIST).
    * Gelu tested against Relu & Elu.
    * Gelu did substantially better.
    * Performs better with dropout
      * Better with noisy data
    * Hendrycks, D., & Gimpel, K. (2016). Gaussian error    linear units (gelus). arXiv preprint arXiv:1606.08415.
    [Link to reference](https://arxiv.org/abs/1606.08415)
* **Analysis of Results**
  * Add all graphs, find average accuracy
  * Analyse the effect on:
    * Epoch Count
    * Learning rate value
  * Analysis of confusion matrix
    * Any evidence of overfitting?

architecture 4,8,16,32,32,64,64,32,32,16,8,4,3 = 94.7%

architecture 4,8,16,32,32,16,8,4,3 = 95.3%

architecture 4,8,16,8,4,3 = 94%

* **Evaluation of NN**
  * How do the results compare to the experiment 'hypothesis'
  * Future research and improvements
