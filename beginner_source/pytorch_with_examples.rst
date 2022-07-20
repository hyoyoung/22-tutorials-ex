예제로 배우는 파이토치(PyTorch)
******************************
**Author**: `Justin Johnson <https://github.com/jcjohnson/pytorch-examples>`_
**번역**: `심형준`

.. 메모::
	이 문서는 오래된 파이토치 튜토리얼 중 하나입니다. 입문자를 위한 최신 문서는 
	`기초 익히기 <https://pytorch.org/tutorials/beginner/basics/intro.html>`_에서 보실 수 있습니다.

이 튜토리얼은 `파이토치(PyTorch) <https://github.com/pytorch/pytorch>`_의 핵심적인 개념을 
예제를 통해 소개합니다.

파이토치(PyTorch)는 핵심적으로 두 가지 주요 기능을 제공합니다.:

- 넘파이(NumPy)와 유사하지만 GPU에서 실행할 수 있는 n-차원 텐서(Tensor)
- 신경망 구축 및 학습을 위한 자동 미분(Automatic differentiation) 
지금부터 3차 다항식(third order polynomial)을 이용해 :math:`y=\sin(x)` 에 근사(fit)하는 예제를 다뤄보겠습니다.
The network will have four parameters, and will be trained with
신경망은 4개의 파라미터(parameters)를 가지며, 정답과 신경망의 예측치 사이의 유클리드 거리(Euclidean distance)를 최소화해 임의의 값을 근사할 수 있도록
경사하강법(gradient descent)을 이용해 학습하겠습니다.

.. 메모::
	각 예제들은
	:ref:`end of this page <examples-download>`에서 살펴볼 수 있습니다.

.. contents:: 목차
	:local:

텐서(Tensors)
=======

워밍업: 넘파이(NumPy)
--------------

PyTorch를 소개하기 전에, NumPy를 이용하여 신경망을 구현해 보겠습니다.

NumPy는 n-차원 배열 객체와 이러한 배열들을 조작하기 위한 다양한 함수를 제공합니다. Numpy는 과학 분야의 연산을 위한 
일반적인 프레임워크(framework)입니다; NumPy는 연산 그래프(computation graphs), 딥러닝, 경사(gradients)는 다루지 않습니다.
그러나 NumPy 연산을 사용하여 신경망의 순전파와 역전파를 수동으로 구현함으로써, 3차 다항식이 sine 함수에
근사하도록 만들 수 있습니다.

.. includenodoc:: /beginner/examples_tensor/polynomial_numpy.py


파이토치(PyTorch): 텐서(Tensors)
----------------

Numpy is a great framework, but it cannot utilize GPUs to accelerate its
numerical computations. For modern deep neural networks, GPUs often
provide speedups of `50x or
greater <https://github.com/jcjohnson/cnn-benchmarks>`__, so
unfortunately numpy won't be enough for modern deep learning.

Here we introduce the most fundamental PyTorch concept: the **Tensor**.
A PyTorch Tensor is conceptually identical to a numpy array: a Tensor is
an n-dimensional array, and PyTorch provides many functions for
operating on these Tensors. Behind the scenes, Tensors can keep track of
a computational graph and gradients, but they're also useful as a
generic tool for scientific computing.

Also unlike numpy, PyTorch Tensors can utilize GPUs to accelerate
their numeric computations. To run a PyTorch Tensor on GPU, you simply
need to specify the correct device.

Here we use PyTorch Tensors to fit a third order polynomial to sine function.
Like the numpy example above we need to manually implement the forward
and backward passes through the network:

.. includenodoc:: /beginner/examples_tensor/polynomial_tensor.py


Autograd
========

파이토치(PyTorch): 텐서(Tensors)와 autograd
-------------------------------

In the above examples, we had to manually implement both the forward and
backward passes of our neural network. Manually implementing the
backward pass is not a big deal for a small two-layer network, but can
quickly get very hairy for large complex networks.

Thankfully, we can use `automatic
differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`__
to automate the computation of backward passes in neural networks. The
**autograd** package in PyTorch provides exactly this functionality.
When using autograd, the forward pass of your network will define a
**computational graph**; nodes in the graph will be Tensors, and edges
will be functions that produce output Tensors from input Tensors.
Backpropagating through this graph then allows you to easily compute
gradients.

This sounds complicated, it's pretty simple to use in practice. Each Tensor
represents a node in a computational graph. If ``x`` is a Tensor that has
``x.requires_grad=True`` then ``x.grad`` is another Tensor holding the
gradient of ``x`` with respect to some scalar value.

Here we use PyTorch Tensors and autograd to implement our fitting sine wave
with third order polynomial example; now we no longer need to manually
implement the backward pass through the network:

.. includenodoc:: /beginner/examples_autograd/polynomial_autograd.py

파이토치(PyTorch): 새로운 autograd 함수 정의하기
----------------------------------------

Under the hood, each primitive autograd operator is really two functions
that operate on Tensors. The **forward** function computes output
Tensors from input Tensors. The **backward** function receives the
gradient of the output Tensors with respect to some scalar value, and
computes the gradient of the input Tensors with respect to that same
scalar value.

In PyTorch we can easily define our own autograd operator by defining a
subclass of ``torch.autograd.Function`` and implementing the ``forward``
and ``backward`` functions. We can then use our new autograd operator by
constructing an instance and calling it like a function, passing
Tensors containing input data.

In this example we define our model as :math:`y=a+b P_3(c+dx)` instead of
:math:`y=a+bx+cx^2+dx^3`, where :math:`P_3(x)=\frac{1}{2}\left(5x^3-3x\right)`
is the `Legendre polynomial`_ of degree three. We write our own custom autograd
function for computing forward and backward of :math:`P_3`, and use it to implement
our model:

.. _Legendre polynomial:
    https://en.wikipedia.org/wiki/Legendre_polynomials

.. includenodoc:: /beginner/examples_autograd/polynomial_custom_function.py

`nn` 모듈(module)
===========

파이토치(PyTorch): nn
-----------

Computational graphs and autograd are a very powerful paradigm for
defining complex operators and automatically taking derivatives; however
for large neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the
computation into **layers**, some of which have **learnable parameters**
which will be optimized during learning.

In TensorFlow, packages like
`Keras <https://github.com/fchollet/keras>`__,
`TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__,
and `TFLearn <http://tflearn.org/>`__ provide higher-level abstractions
over raw computational graphs that are useful for building neural
networks.

In PyTorch, the ``nn`` package serves this same purpose. The ``nn``
package defines a set of **Modules**, which are roughly equivalent to
neural network layers. A Module receives input Tensors and computes
output Tensors, but may also hold internal state such as Tensors
containing learnable parameters. The ``nn`` package also defines a set
of useful loss functions that are commonly used when training neural
networks.

In this example we use the ``nn`` package to implement our polynomial model
network:

.. includenodoc:: /beginner/examples_nn/polynomial_nn.py

파이토치(PyTorch): optim
--------------

Up to this point we have updated the weights of our models by manually
mutating the Tensors holding learnable parameters with ``torch.no_grad()``.
This is not a huge burden for simple optimization algorithms like stochastic
gradient descent, but in practice we often train neural networks using more
sophisticated optimizers like AdaGrad, RMSProp, Adam, etc.

The ``optim`` package in PyTorch abstracts the idea of an optimization
algorithm and provides implementations of commonly used optimization
algorithms.

In this example we will use the ``nn`` package to define our model as
before, but we will optimize the model using the RMSprop algorithm provided
by the ``optim`` package:

.. includenodoc:: /beginner/examples_nn/polynomial_optim.py

파이토치(PyTorch): 사용자 정의 nn.Modules
--------------------------

Sometimes you will want to specify models that are more complex than a
sequence of existing Modules; for these cases you can define your own
Modules by subclassing ``nn.Module`` and defining a ``forward`` which
receives input Tensors and produces output Tensors using other
modules or other autograd operations on Tensors.

In this example we implement our third order polynomial as a custom Module
subclass:

.. includenodoc:: /beginner/examples_nn/polynomial_module.py

파이토치(PyTorch): 흐름 제어(Control Flow) + 가중치 공유(Weight Sharing)
--------------------------------------

As an example of dynamic graphs and weight sharing, we implement a very
strange model: a third-fifth order polynomial that on each forward pass
chooses a random number between 3 and 5 and uses that many orders, reusing
the same weights multiple times to compute the fourth and fifth order.

For this model we can use normal Python flow control to implement the loop,
and we can implement weight sharing by simply reusing the same parameter multiple
times when defining the forward pass.

We can easily implement this model as a Module subclass:

.. includenodoc:: /beginner/examples_nn/dynamic_net.py


.. _examples-download:

예제
========

You can browse the above examples here.

텐서(Tensors)
-------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_tensor/polynomial_numpy
   /beginner/examples_tensor/polynomial_tensor

.. galleryitem:: /beginner/examples_tensor/polynomial_numpy.py

.. galleryitem:: /beginner/examples_tensor/polynomial_tensor.py

.. raw:: html

    <div style='clear:both'></div>

Autograd
--------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_autograd/polynomial_autograd
   /beginner/examples_autograd/polynomial_custom_function


.. galleryitem:: /beginner/examples_autograd/polynomial_autograd.py

.. galleryitem:: /beginner/examples_autograd/polynomial_custom_function.py

.. raw:: html

    <div style='clear:both'></div>

`nn` module
-----------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_nn/polynomial_nn
   /beginner/examples_nn/polynomial_optim
   /beginner/examples_nn/polynomial_module
   /beginner/examples_nn/dynamic_net


.. galleryitem:: /beginner/examples_nn/polynomial_nn.py

.. galleryitem:: /beginner/examples_nn/polynomial_optim.py

.. galleryitem:: /beginner/examples_nn/polynomial_module.py

.. galleryitem:: /beginner/examples_nn/dynamic_net.py

.. raw:: html

    <div style='clear:both'></div>
