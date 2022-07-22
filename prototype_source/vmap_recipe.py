"""
torch.vmap
==========
이 튜토리얼에서는 PyTorch 작업을 위한 자동 벡터라이저인 torch.vmap을 소개합니다. 
torch.vmap은 프로토타입 기능이기 때문에 많은 사용 사례를 처리할 수 없지만, 
디자인을 알리기 위해 사용 사례를 수집하고자 합니다. torch.vmap 사용을 
고려 중이거나 정말 멋진 것이라고 생각한다면
https://github.com/pytorch/pytorch/issues/42368으로 문의하세요.

vmap이 뭔가요?
--------------
vmap은 고차 함수입니다. 함수 `func` 를 받고, 입력의 일부 차원에 `func` 를 매핑하는 
새로운 함수를 반환합니다. JAX의 vmap에서 크게 영감을 받았습니다.

의미적으로 vmap은 `func` 에 의해 호출된 PyTorch 작업에 "map"을 푸시하여 해당 
작업을 효과적으로 벡터화합니다.
"""
import torch
# NB: vmap은 PyTorch의 nightly 빌드 버전에서만 사용할 수 있습니다. 
# 테스트하고 싶다면 pytorch.org에서 다운로드할 수 있습니다.
from torch import vmap

####################################################################
# vmap의 첫 번째 사용 사례는 코드에서 배치 차원을 더 쉽게 처리할 수 있도록
# 하는 것입니다. 예제에서처럼 함수 `func` 를 작성한 후, `vmap(func)` 와 같이
# 일괄 처리 함수로 사용할 수 있습니다. 그러나 `func` 에는 많은 제한이 있습니다.
#
# - 내부 PyTorch 작업을 제외하고 기능적이어야 합니다(Python 데이터 구조를 변경할
#   수 없음).
#
# - 배치는 반드시 Tensor로 제공되어야 합니다. 즉, vmap은 가변 길이 시퀀스를 즉시 
#   처리하지 않습니다.
#
# `vmap` 을 사용하는 한 가지 예는 내적을 배치로 계산하는 것입니다. PyTorch는 배치된 
# `torch.dot` API를 제공하지 않습니다. 문서를 헤매는 대신 `vmap` 을 사용하여 새로운
# 함수를 구성하세요.

torch.dot                            # [D], [D] -> []
batched_dot = torch.vmap(torch.dot)  # [N, D], [N, D] -> [N]
x, y = torch.randn(2, 5), torch.randn(2, 5)
batched_dot(x, y)

####################################################################
# `vmap` can be helpful in hiding batch dimensions, leading to a simpler
# model authoring experience.
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)

# Note that model doesn't work with a batch of feature vectors because
# torch.dot must take 1D tensors. It's pretty easy to rewrite this
# to use `torch.matmul` instead, but if we didn't want to do that or if
# the code is more complicated (e.g., does some advanced indexing
# shenanigins), we can simply call `vmap`. `vmap` batches over ALL
# inputs, unless otherwise specified (with the in_dims argument,
# please see the documentation for more details).
def model(feature_vec):
    # Very simple linear model with activation
    return feature_vec.dot(weights).relu()

examples = torch.randn(batch_size, feature_size)
result = torch.vmap(model)(examples)
expected = torch.stack([model(example) for example in examples.unbind()])
assert torch.allclose(result, expected)

####################################################################
# `vmap` can also help vectorize computations that were previously difficult
# or impossible to batch. This bring us to our second use case: batched
# gradient computation.
#
# - https://github.com/pytorch/pytorch/issues/8304
# - https://github.com/pytorch/pytorch/issues/23475
#
# The PyTorch autograd engine computes vjps (vector-Jacobian products).
# Using vmap, we can compute (batched vector) - jacobian products.
#
# One example of this is computing a full Jacobian matrix (this can also be
# applied to computing a full Hessian matrix).
# Computing a full Jacobian matrix for some function f: R^N -> R^N usually
# requires N calls to `autograd.grad`, one per Jacobian row.

# Setup
N = 5
def f(x):
    return x ** 2

x = torch.randn(N, requires_grad=True)
y = f(x)
basis_vectors = torch.eye(N)

# Sequential approach
jacobian_rows = [torch.autograd.grad(y, x, v, retain_graph=True)[0]
                 for v in basis_vectors.unbind()]
jacobian = torch.stack(jacobian_rows)

# Using `vmap`, we can vectorize the whole computation, computing the
# Jacobian in a single call to `autograd.grad`.
def get_vjp(v):
    return torch.autograd.grad(y, x, v)[0]

jacobian_vmap = vmap(get_vjp)(basis_vectors)
assert torch.allclose(jacobian_vmap, jacobian)

####################################################################
# The third main use case for vmap is computing per-sample-gradients.
# This is something that the vmap prototype cannot handle performantly
# right now. We're not sure what the API for computing per-sample-gradients
# should be, but if you have ideas, please comment in
# https://github.com/pytorch/pytorch/issues/7786.

def model(sample, weight):
    # do something...    
    return torch.dot(sample, weight)

def grad_sample(sample):
    return torch.autograd.functional.vjp(lambda weight: model(sample), weight)[1]

# The following doesn't actually work in the vmap prototype. But it
# could be an API for computing per-sample-gradients.

# batch_of_samples = torch.randn(64, 5)
# vmap(grad_sample)(batch_of_samples)
