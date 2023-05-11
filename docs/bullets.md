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
* 
