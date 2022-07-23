안드로이드를 위한 모델 준비 레시피
=====================================

이 레시피는 안드로이드 앱용 파이토치 MobileNet v2 이미지 분류 모델을 준비하는 방법과 모바일 지원 가능한 모델 파일을 사용할 수 있도록 안드로이드 프로젝트를 설정하는 방법에 대해 설명합니다.

소개
-----------------

파이토치 모델이 학습되거나 사전 학습된 모델을 사용할 수 있게 된 후에는 일반적으로 아직 모바일 앱에서 사용할 준비가 되어 있지 않습니다. 양자화(`양자화 레시피 <quantization.rst>`_ 참조)하여 TorchScript로 변환하여 안드로이드 앱이 모델을 불러올 수 있도록 하고 모바일 앱에 최적화해야 합니다. 또한, 안드로이드 앱이 모델을 불러와서 추론을 위해 사용하기 전에 PyTorch Mobile 라이브러리를 사용할 수 있도록 안드로이드 앱을 올바르게 설정해야 합니다.

요구사항(Pre-requisites)
-----------------

PyTorch 1.6.0 혹은 1.7.0

torchvision 0.6.0 혹은 0.7.0

NDK가 설치된 Android Studio 3.5.1 혹은 더 상위 버전

방법
-----------------

1. 사전 학습 및 양자화된 MobileNet v2 모델 가져오기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

사전 학습 및 양자화된 MobileNet v2 모델 가져오기 위해서는 간단하게 아래의 코드를 실행합니다:
::

    import torchvision

    model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)

2. 모바일 앱을 위한 모델 스크립팅 및 최적화
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

'script' 또는 'trace' 방법을 사용하여 양자화된 모델을 TorchScript 형식으로 변환합니다:

::

    import torch

    dummy_input = torch.rand(1, 3, 224, 224)
    torchscript_model = torch.jit.trace(model_quantized, dummy_input)

또는

::

    torchscript_model = torch.jit.script(model_quantized)


경고

trace 메서드는 추적 중에 실행되는 코드 경로만 스크립팅하므로 의사결정 분기가 포함된 모델에서는 제대로 작동하지 않습니다. 
자세한 내용은 `모바일을 위한 스트립팅과 최적화 레시피 <script_optimized.rst>`_ 을 참조하십시오.
    
그런 다음 TorchScript 형식 모델을 모바일용으로 최적화하고 저장합니다:

::

    from torch.utils.mobile_optimizer import optimize_for_mobile
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized, "mobilenetv2_quantized.pt")
    
위의 두 단계에서 총 7줄 또는 8줄의 코드(모델의 TorchScript 형식을 가져오기 위해 'script' 또는 'trace' 메서드가 호출되는지에 따라)로, 우리는 모바일 앱에 추가할 수 있는 모델을 준비했습니다.

3. 안드로이드에서 모델 및 PyTorch 라이브러리 추가
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* 현재 또는 새 Android Studio 프로젝트에서 build.gradle 파일을 열고 다음 두 줄을 추가합니다(두 번째 줄은 TorchVision 모델을 사용하려는 경우에만 필요합니다).

::

    implementation 'org.pytorch:pytorch_android:1.6.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.6.0'

* 모델 파일 'mobilenetv2_quantized.pt'를 프로젝트의 자산 폴더로 드래그 앤 드롭합니다.

끝났습니다! 이제 PyTorch 라이브러리와 준비된 모델로 안드로이드 앱을 만들 수 있습니다. 실제로 모델을 사용하기 위한 코드를 작성하려면 PyTorch Mobile `안드로이드 퀵스타트 with HelloWorld 예제 <https://pytorch.org/mobile/android/#quickstart-with-a-helloworld-example>`_ 와 `안드로이드 해커톤 예제 <https://github.com/pytorch/workshops/tree/master/PTMobileWalkthruAndroid>`_ 를 참조하십시오.

더 알아보기
-----------------

1. `PyTorch Mobile site <https://pytorch.org/mobile>`_

2. `Introduction to TorchScript <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_
