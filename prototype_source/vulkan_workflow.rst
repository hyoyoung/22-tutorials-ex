PyTorch Vulkan 백엔드 사용자 워크플로
============================================

**Author**: `Ivan Kobzarev <https://github.com/IvanKobzarev>`_

소개
------
PyTorch 1.7은 Vulkan 그래픽과 컴퓨팅 API를 지원하는 GPU에서 모델 추론(inference)을 실행하는 것을 지원합니다. 주 목표 디바이스는 안드로이드 장치의 모바일 GPU입니다. Vulkan 백엔드는 Vulkan을 지원하는 Intel 통합 GPU와 같은 장치를 사용해 Linux, Mac 그리고 Windows 데스크톱 빌드에서도 사용할 수 있습니다. 이 기능은 프로토타입 단계이며 변경될 수 있습니다.

PyTorch에서 Vulkan 백엔드 빌드하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vulkan 백엔드는 기본적으로 포함되어 있지 않습니다. Vulkan 백엔드를 포함하는 cmake 옵션은 ``USE_VULKAN`` 이고, 환경 변수 ``USE_VULKAN`` 에 의해 설정될 수 있습니다.

PyTorch에서 Vulkan 백엔드를 사용하기 위해서는, 소스에서 추가적인 설정을 하여 빌드해야 합니다. PyTorch 소스 코드를 Github master 브랜치에서 확인하세요.

Vulkan 래퍼(wrapper)의 선택적 사용
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

기본적으로, Vulkan 라이브러리는 런타임이 vulkan_wrapper 라이브러리를 사용할 때 로딩됩니다. 만약 ``USE_VULKAN_WRAPPER=0`` 으로 환경 변수를 특정한다면 libvulkan이 직접 연결됩니다.

데스크톱 빌드
^^^^^^^^^^^^^^^^^

Vulkan SDK
^^^^^^^^^^
VulkanSDK를 https://vulkan.lunarg.com/sdk/home 에서 다운로드 하고 ``VULKAN_SDK`` 환경 변수를 설정합니다.

VulkanSDK를 ``VULKAN_SDK_ROOT`` 폴더에 압축 해제하고, 아래 VulkanSDK 설치법을 따라 VulkanSDK를 시스템에 설치합니다.

Mac용:

::

    cd $VULKAN_SDK_ROOT
    source setup-env.sh
    sudo python install_vulkan.py


PyTorch 빌드하기:

Linux용:

::

    cd PYTORCH_ROOT
    USE_VULKAN=1 USE_VULKAN_SHADERC_RUNTIME=1 USE_VULKAN_WRAPPER=0 python setup.py install

Mac용:

::

    cd PYTORCH_ROOT
    USE_VULKAN=1 USE_VULKAN_SHADERC_RUNTIME=1 USE_VULKAN_WRAPPER=0 MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

성공적으로 빌드한 이후, 다른 터미널 창을 열어 설치된 PyTorch 버전을 확인합니다.

::

    import torch
    print(torch.__version__)

지금 이 튜토리얼을 작성하는 기준으로, 버전은 1.8.0a0+41237a4입니다. Master 코드를 확인한 시점에 따라 다른 숫자를 볼 수도 있지만, 1.7.0 이상이어야 합니다.


Android 빌드
^^^^^^^^^^^^^

To build LibTorch for android with Vulkan backend for specified ``ANDROID_ABI``.

::

    cd PYTORCH_ROOT
    ANDROID_ABI=arm64-v8a USE_VULKAN=1 sh ./scripts/build_android.sh


To prepare pytorch_android aars that you can use directly in your app:

::

    cd $PYTORCH_ROOT
    USE_VULKAN=1 sh ./scripts/build_pytorch_android.sh


Model preparation
-----------------

Install torchvision, get the default pretrained float model.

::

    pip install torchvision

Python script to save pretrained mobilenet_v2 to a file:

::

    import torch
    import torchvision

    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.eval()
    script_model = torch.jit.script(model)
    torch.jit.save(script_model, "mobilenet2.pt")

PyTorch 1.7 Vulkan backend supports only float 32bit operators. The default model needs additional step that will optimize operators fusing 

::

    from torch.utils.mobile_optimizer import optimize_for_mobile
    script_model_vulkan = optimize_for_mobile(script_model, backend='vulkan')
    torch.jit.save(script_model_vulkan, "mobilenet2-vulkan.pt")

The result model can be used only on Vulkan backend as it contains specific to the Vulkan backend operators.

Using Vulkan backend in code
----------------------------

C++ API
-------

::

    at::is_vulkan_available()
    auto tensor = at::rand({1, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
    auto tensor_vulkan = t.vulkan();
    auto module = torch::jit::load("$PATH");
    auto tensor_output_vulkan = module.forward(inputs).toTensor();
    auto tensor_output = tensor_output.cpu();

``at::is_vulkan_available()`` function tries to initialize Vulkan backend and if Vulkan device is successfully found and context is created - it will return true, false otherwise.

``.vulkan()`` function called on Tensor will copy tensor to Vulkan device, and for operators called with this tensor as input - the operator will run on Vulkan device, and its output will be on the Vulkan device.

``.cpu()`` function called on Vulkan tensor will copy its data to CPU tensor (default)

Operators called with a tensor on a Vulkan device as an input will be executed on a Vulkan device. If an operator is not supported for the Vulkan backend the exception will be thrown.

List of supported operators:

::

    _adaptive_avg_pool2d
    _cat
    add.Scalar
    add.Tensor
    add_.Tensor
    addmm
    avg_pool2d
    clamp
    convolution
    empty.memory_format
    empty_strided
    hardtanh_
    max_pool2d
    mean.dim
    mm
    mul.Scalar
    relu_
    reshape
    select.int
    slice.Tensor
    transpose.int
    transpose_
    unsqueeze
    upsample_nearest2d
    view

Those operators allow to use torchvision models for image classification on Vulkan backend.


Python API
----------

``torch.is_vulkan_available()`` is exposed to Python API.

``tensor.to(device='vulkan')`` works as ``.vulkan()`` moving tensor to the Vulkan device.

``.vulkan()`` at the moment of writing of this tutorial is not exposed to Python API, but it is planned to be there.

Android Java API
---------------

For Android API to run model on Vulkan backend we have to specify this during model loading:

::

    import org.pytorch.Device;
    Module module = Module.load("$PATH", Device.VULKAN)
    FloatBuffer buffer = Tensor.allocateFloatBuffer(1 * 3 * 224 * 224);
    Tensor inputTensor = Tensor.fromBlob(buffer, new int[]{1, 3, 224, 224});
    Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();

In this case, all inputs will be transparently copied from CPU to the Vulkan device, and model will be run on Vulkan device, the output will be copied transparently to CPU.

The example of using Vulkan backend can be found in test application within the PyTorch repository:
https://github.com/pytorch/pytorch/blob/master/android/test_app/app/src/main/java/org/pytorch/testapp/MainActivity.java#L133

Building android test app with Vulkan
-------------------------------------

1. Build pytorch android with Vulkan backend for all android ABIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    cd $PYTORCH_ROOT
    USE_VULKAN=1 sh ./scripts/build_pytorch_android.sh

Or if you need only specific abi you can set it as an argument:

::

    cd $PYTORCH_ROOT
    USE_VULKAN=1 sh ./scripts/build_pytorch_android.sh $ANDROID_ABI

2. Add vulkan model to test application assets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add prepared model ``mobilenet2-vulkan.pt`` to test applocation assets:

::
  
  cp mobilenet2-vulkan.pt $PYTORCH_ROOT/android/test_app/app/src/main/assets/


3. Build and Install test applocation to connected android device 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    cd $PYTORCH_ROOT
    gradle -p android test_app:installMbvulkanLocalBaseDebug

After successful installation, the application with the name 'MBQ' can be launched on the device. 





Testing models without uploading to android device
--------------------------------------------------

Software implementations of Vulkan (e.g. https://swiftshader.googlesource.com/SwiftShader ) can be used to test if a model can be run using PyTorch Vulkan Backend (e.g. check if all model operators are supported).
