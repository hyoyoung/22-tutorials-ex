양자화 사용법(Quantization Recipe)
=====================================

이 사용법(recipe)는 파이토치(Pytorch) 모델을 양자화하는 방법을 설명한다. 양자화된 모델은 원본 모델과 거의 같은 정확도를 내면서, 사이즈가 줄고 추론 속도가 빨라진다. 양자화 작업은 서버 모델과 모바일 모델 배포 모두에 적용될 수 있지만, 모바일 환경에서 특히 중요하고 매우 필요하다. 그 이유는 양자화를 적용하지 않은 모델의 크기가 IOS나 Android 앱이 허용하는 크기 한도를 초과하고, 모델의 배포나 OTA 업데이트 시간이 너무 오래 잡아먹으며, 또한 추론 속도가 너무 오래 걸려서 사용자들의 쾌적함을 방해하기 때문이다.

소개(Introduction)
------------

양자화는 모델 파라미터를 구성하는 32-비트 크기의 실수 자료형의 숫자를 8-비트 크기의 정수 자료형의 숫자로 전환하는 기법이다. 양자화 기법을 적용하면, 정확도는 거의 같게 유지하면서, 모델의 크기와 memory footprint를 원본 모델의 4분의 1까지 감소시킬 수 있고, 추론은 2-4배 정도 빠르게 만들 수 있다. 

모델을 양자화하는 데는 전부 세 가지의 접근법 및 작업 흐름(workflows)이 있다. : 학습 후 동적(dynamic) 양자화, 학습 후 정적(static) 양자화, 그리고 양자화를 고려하는 학습법이 있습니다. 하지만 사용하려는 모델이 이미 양자화된 버전이 있다면, 위의 세 가지 방식을 거치지 않고 그 버전을 바로 사용하면 됩니다. 예를 들어, torchvision 라이브러리에는 이미 다음 모델들의 양자화된 버전이 존재합니다 : MobileNet v2, ResNet 18, ResNet 50, Inception v3, GoogleNet, 등. 따라서 비록 단순한 작업이겠지만, 이미 존재하는 모델을 사용하는 방법을 또다른 작업 방식 중 하나로 포함하려 합니다.

.. note::
    양자화는 일부 제한된 범위의 연산자에서만 지원이 가능합니다. 더 많은 정보는 `여기 <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#device-and-operator-support>`_ 를 참고하세요.

필요 조건(Pre-requisites)
-----------------

PyTorch 1.6.0 or 1.7.0

torchvision 0.6.0 or 0.7.0

작업 방식 (Workflows)
------------

양자화를 진행하려면 다음 4가지 방식 중 하나를 사용하세요.

1. 사전 학습 및 양자화된 MobileNet v2 사용하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

사전 학습된 MobileNet v2 모델을 불러오려면, 다음을 입력하세요 :

::

    import torchvision
    model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)


양자화 전의 모델과 양자화된 버전의 MobileNet v2 모델의 크기를 비교하려면 : 

::

    model = torchvision.models.mobilenet_v2(pretrained=True)

    import os
    import torch

    def print_model_size(mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
        os.remove('tmp.pt')

    print_model_size(model)
    print_model_size(model_quantized)


결과는 다음과 같습니다. :

::

    14.27 MB
    3.63 MB

2. 학습 후 동적(Dynamic) 양자화 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

동적(Dynamic) 양자화를 적용하면, 모델의 모든 가중치(weights)는 32-비트 크기의 실수 자료형에서 8-비트 크기의 정수 자료형으로 전환되지만, 활성화에 대한 계산을 진행하기 직전까지는 활성 함수는 8-비트 정수형으로 전환하지 않게 됩니다. 동적(Dynamic) 양자화를 적용하려면, `torch.quantization.quantize_dynamic` 을 사용하면 됩니다: 

::

    model_dynamic_quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )

여기서 `qconfig_spec` 은 `model` 내에서 양자화 적용 대상인 내부 모듈(submodules)을 특정짓습니다.

.. warning:: 동적(Dynamic) 양자화는 사전-학습된 양자화 적용 모델이 준비되지 않았을 때 사용하기 가장 쉬운 방식이지만, 이 방식의 주요 한계는 qconfig_spec 옵션이 현재는 nn.Linear과 nn.LSTM만 지원한다는 것입니다. 이는 당신이 nn.Conv2d와 같은 다른 모듈을 양자화할 때, 나중에 논의될 정적(Static) 양자화나 양자화를 고려하는 학습법을 사용해야 한다는 걸 의미합니다.

quantize_dynamic API call 관련 원본 문서는 `여기 <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`_를 참고하세요. 학습 후 동적(Dynamic) 양자화를 사용하는 세 가지 예제에는 `the Bert example <https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html>`_, `an LSTM model example <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html#test-dynamic-quantization>`_, 그리고 또 `demo LSTM example <https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html#do-the-quantization>`_이 있습니다.

3. Post Training Static Quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method converts both the weights and the activations to 8-bit integers beforehand so there won't be on-the-fly conversion on the activations during the inference, as the dynamic quantization does, hence improving the performance significantly.

To apply static quantization on a model, run the following code:

::

    backend = "qnnpack"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

After this, running `print_model_size(model_static_quantized)` shows the static quantized model is `3.98MB`.

A complete model definition and static quantization example is `here <https://pytorch.org/docs/stable/quantization.html#quantization-api-summary>`_. A dedicated static quantization tutorial is `here <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.

.. note::
  To make the model run on mobile devices which normally have arm architecture, you need to use `qnnpack` for `backend`; to run the model on computer with x86 architecture, use `fbgemm`.

4. Quantization Aware Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantization aware training inserts fake quantization to all the weights and activations during the model training process and results in higher inference accuracy than the post-training quantization methods. It is typically used in CNN models.

To enable a model for quantization aware traing, define in the `__init__` method of the model definition a `QuantStub` and a `DeQuantStub` to convert tensors from floating point to quantized type and vice versa:

::

    self.quant = torch.quantization.QuantStub()
    self.dequant = torch.quantization.DeQuantStub()

Then in the beginning and the end of the `forward` method of the model definition, call `x = self.quant(x)` and `x = self.dequant(x)`.

To do a quantization aware training, use the following code snippet:

::

    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    model_qat = torch.quantization.prepare_qat(model, inplace=False)
    # quantization aware training goes here
    model_qat = torch.quantization.convert(model_qat.eval(), inplace=False)

For more detailed examples of the quantization aware training, see `here <https://pytorch.org/docs/master/quantization.html#quantization-aware-training>`_ and `here <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training>`_.

A pre-trained quantized model can also be used for quantized aware transfer learning, using the same `quant` and `dequant` calls shown above. See `here <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html#part-1-training-a-custom-classifier-based-on-a-quantized-feature-extractor>`_ for a complete example.

After a quantized model is generated using one of the steps above, before the model can be used to run on mobile devices, it needs to be further converted to the `TorchScript` format and then optimized for mobile apps. See the `Script and Optimize for Mobile recipe <script_optimized.html>`_ for details.

Learn More
-----------------

For more info on the different workflows of quantization, see `here <https://pytorch.org/docs/stable/quantization.html#quantization-workflows>`_ and `here <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#post-training-static-quantization>`_.
