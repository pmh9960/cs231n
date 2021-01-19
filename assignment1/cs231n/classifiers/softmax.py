import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_examples, n_feature = X.shape
  for i in range(n_examples):
    denominator = np.sum(np.exp(np.dot(X[i], W)))
    numerator = np.exp(np.dot(X[i], W))
    softmax_c = numerator / denominator
    loss += - np.log(softmax_c[y[i]])

    grad = softmax_c * np.expand_dims(X[i], axis=1)
    grad.T[y[i]] = -(1 - softmax_c[y[i]]) * X[i]
    dW += grad
  loss /= n_examples
  loss += reg * np.sum(W * W)

  dW /= n_examples
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_examples, n_feature = X.shape
  denominator = np.sum(np.exp(np.dot(X, W)), axis=1, keepdims=True)
  numerator = np.exp(np.dot(X, W))
  softmax_c = numerator / denominator
  idx = np.arange(y.shape[0])
  loss = - np.log(softmax_c[idx, y])
  loss = np.sum(loss) / n_examples
  loss += reg * np.sum(W * W)

  grad = np.expand_dims(softmax_c, axis=2) * np.expand_dims(X, axis=1)
  grad[idx, y] = -np.expand_dims((1 - softmax_c[idx, y]), axis=1) * X[idx]
  dW = np.sum(grad, axis=0).T / n_examples
  dW += reg * 2 * W

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

