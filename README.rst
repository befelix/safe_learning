=====================================================
Safe Reinforcement Learning with Stability Guarantees
=====================================================

.. image:: https://travis-ci.org/befelix/safe_learning.svg?branch=master
    :target: https://travis-ci.org/befelix/safe_learning
    :alt: Build status
.. image:: https://readthedocs.org/projects/safe-learning/badge/?version=latest
    :target: http://safe-learning.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

This code accompanies the paper [1]_ and implements the code for estimating the region of attraction for a policy and optimizing the policy subject to stability constraints. For the old numpy-based code to estimate the region of attraction in [2]_ see the `lyapunov-learning <https://github.com/befelix/lyapunov-learning>`_ repository.

.. [1] F. Berkenkamp, M. Turchetta, A. P. Schoellig, A. Krause,
  `Safe Model-based Reinforcement Learning with Stability Guarantees <http://arxiv.org/abs/1509.01066>`_
  in Proc. of the Conference on Neural Information Processing Systems (NIPS), 2017.
  
.. [2] F. Berkenkamp, R. Moriconi, A. P. Schoellig, A. Krause, 
  `Safe Learning of Regions of Attraction in Uncertain, Nonlinear Systems with Gaussian Processes` <http://arxiv.org/abs/1603.04915>
  in Proc. of the Conference on Decision and Control (CDC), 2016.

Getting started
---------------

You can install the library by cloning the repository and installing it with

``pip install .``

You can the find example jupyter notebooks and the experiments in the paper in the `examples <./examples>`_ folder.

