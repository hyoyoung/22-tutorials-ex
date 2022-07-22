(prototype)  추적-기반 선택적 빌드 모바일 안드로이드,IOS 인터프리터
===============================================================================


*Author*: Chen Lai <https://github.com/cccclai>, Dhruv Matani <https://github.com/dhruvbird>

.. 경고::
추적-기반 선택 빌드는 라이브러리 사이즈를 줄이기 위한 프로토타입 기능입니다. 추적된 결과는 모델 입력과 추적된 환경에 의존 하기때문에, 
만약 추적기가 모바일 인터프리터가 아닌 다른 환경에서 실행하면, 작동자 리스트가 실제 사용된 작동자 리스트와 다를 수 있고 빠진 작동자들이 오류를 발생 시킬수 있습니다.

Introduction
------------

이 튜토리얼은 모바일 인터프리터 사이즈를 더욱 더 최소화 하기위해, 모바일 인터프리터 빌드를 커스터마이즈하는 새로운 방법을 소개합니다. 이것은 컴파일된 이진 파일에 포함되는 작동자들을 목표 모형에 실제로 필요로하는 작동자 집합 만큼으로 제한합니다. 파이토치의 모바일 배포의 이진 파일 사이즈를 줄이는 기술입니다. 추적 기반 선택적 빌드는 모형을 특정 대표 입력값들과 함께 동작시킵니다, 그리고 어떤 작동자들이 호출되었는지 기록합니다. 그러면 빌드는 그 작동자들만 포함하게됩니다. 



다음은  추적-기반 선택적 방법으로 커스텀 모바일 인터프리터를 제작하는 과정들입니다.

1. *인풋들과 모델을 준비한다.*

.. code:: python

    import numpy as np
    import torch
    import torch.jit
    import torch.utils
    import torch.utils.bundled_inputs
    from PIL import Image
    from torchvision import transforms

    # Step 1. Get the model
    model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    scripted_module = torch.jit.script(model)
    # Export full jit version model (not compatible lite interpreter), leave it here for comparison
    scripted_module.save("deeplabv3_scripted.pt")
    # Export lite interpreter version model (compatible with lite interpreter)
    # path = "<base directory where models are stored>"

    scripted_module._save_for_lite_interpreter(f"${path}/deeplabv3_scripted.ptl")

    model_file = f"${path}/deeplabv3_scripted.ptl"

    # Step 2. Prepare inputs for the model
    input_image_1 = Image.open(f"${path}/dog.jpg")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor_1 = preprocess(input_image_1)
    input_batch_1 = input_tensor_1.unsqueeze(0) # create a mini-batch as expected by the model

    scripted_module = torch.jit.load(model_file)
    scripted_module.forward(input_batch_1) # optional, to validate the model can run with the input_batch_1

    input_image_2 = Image.open(f"${path}/deeplab.jpg")
    input_tensor_2 = preprocess(input_image_2)
    input_batch_2 = input_tensor_2.unsqueeze(0) # create a mini-batch as expected by the model

    scripted_module = torch.jit.load(model_file)
    scripted_module.forward(input_batch_2) # optional, to validate the model can run with the input_batch_2

    # Step 3. Bundle the model with the prepared input from step2. Can bundle as many input as possible.
    bundled_model_input = [
        (torch.utils.bundled_inputs.bundle_large_tensor(input_batch_1), ),
        (torch.utils.bundled_inputs.bundle_large_tensor(input_batch_2), )]
    bundled_model = torch.utils.bundled_inputs.bundle_inputs(scripted_module, bundled_model_input)
    bundled_model._save_for_lite_interpreter(f"${path}/deeplabv3_scripted_with_bundled_input.ptl")

2. 추적기를 제작한다.

.. code:: shell

 MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ MAX_JOBS=16 TRACING_BASED=1 python setup.py develop

3. 추적기와 모델을 입력과 함께 동작시키다.

.. code:: shell

 ./build/bin/model_tracer --model_input_path ${path}/deeplabv3_scripted_with_bundled_input.ptl --build_yaml_path ${path}/deeplabv3_scripted.yaml



안드로이드
-------

이미지 영역 분할 안드로이드 데모 앱을 가져옵니다 :  https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation

1. **Tracing-based build libtoㄴrch lite for android**: Build libtorch for android for all 4 android abis (``armeabi-v7a``, ``arm64-v8a``, ``x86``, ``x86_64``) by running

.. code-block:: bash

   SELECTED_OP_LIST=${path}/deeplabv3_scripted.yaml TRACING_BASED=1  ./scripts/build_pytorch_android.sh

if it will be tested on Pixel 4 emulator with ``x86``, use cmd ``BUILD_LITE_INTERPRETER=1 ./scripts/build_pytorch_android.sh x86`` to specify abi to save build time.

.. code-block:: bash

   SELECTED_OP_LIST=${path}/deeplabv3_scripted.yaml TRACING_BASED=1  ./scripts/build_pytorch_android.sh x86


After the build finish, it will show the library path:

.. code-block:: bash

   BUILD SUCCESSFUL in 55s
   134 actionable tasks: 22 executed, 112 up-to-date
   + find /Users/chenlai/pytorch/android -type f -name '*aar'
   + xargs ls -lah
   -rw-r--r--  1 chenlai  staff    13M Feb 11 11:48 /Users/chenlai/pytorch/android/pytorch_android/build/outputs/aar/pytorch_android-release.aar
   -rw-r--r--  1 chenlai  staff    36K Feb  9 16:45 /Users/chenlai/pytorch/android/pytorch_android_torchvision/build/outputs/aar/pytorch_android_torchvision-release.aar

2. **Use the PyTorch Android libraries built from source in the ImageSegmentation app**: Create a folder `libs` in the path, the path from repository root will be `ImageSegmentation/app/libs`. Copy `pytorch_android-release` to the path ``ImageSegmentation/app/libs/pytorch_android-release.aar``. Copy `pytorch_android_torchvision` (downloaded from `Pytorch Android Torchvision Nightly <https://oss.sonatype.org/#nexus-search;quick~torchvision_android/>`_) to the path ``ImageSegmentation/app/libs/pytorch_android_torchvision.aar``. Update the `dependencies` part of ``ImageSegmentation/app/build.gradle`` to

.. code:: gradle

   dependencies {
       implementation 'androidx.appcompat:appcompat:1.2.0'
       implementation 'androidx.constraintlayout:constraintlayout:2.0.2'
       testImplementation 'junit:junit:4.12'
       androidTestImplementation 'androidx.test.ext:junit:1.1.2'
       androidTestImplementation 'androidx.test.espresso:espresso-core:3.3.0'


       implementation(name:'pytorch_android-release', ext:'aar')
       implementation(name:'pytorch_android_torchvision', ext:'aar')

       implementation 'com.android.support:appcompat-v7:28.0.0'
       implementation 'com.facebook.fbjni:fbjni-java-only:0.0.3'
   }

Update `all projects` part in ``ImageSegmentation/build.gradle`` to


.. code:: gradle

    allprojects {
        repositories {
            google()
            jcenter()
            flatDir {
                dirs 'libs'
            }
        }
    }


3. **Test app**: Build and run the `ImageSegmentation` app in Android Studio


iOS
---

Get ImageSegmentation demo app in iOS: https://github.com/pytorch/ios-demo-app/tree/master/ImageSegmentation


1. **Build libtorch lite for iOS**:

.. code-block:: bash

   SELECTED_OP_LIST=${path}/deeplabv3_scripted.yaml TRACING_BASED=1 IOS_PLATFORM=SIMULATOR ./scripts/build_ios.sh


2. **Remove Cocoapods from the project** (this step is only needed if you ran `pod install`):


.. code-block:: bash

   pod deintegrate


3.  **Link ImageSegmentation demo app with the custom built library**:

Open your project in XCode, go to your project Target’s **Build Phases - Link Binaries With Libraries**, click the **+** sign and add all the library files located in `build_ios/install/lib`. Navigate to the project **Build Settings**, set the value **Header Search Paths** to `build_ios/install/include` and **Library Search Paths** to `build_ios/install/lib`.
In the build settings, search for **other linker flags**. Add a custom linker flag below `-all_load`.
Finally, disable bitcode for your target by selecting the Build Settings, searching for Enable Bitcode, and set the value to **No**.


4. **Build and test the app in Xcode.**



Conclusion
----------

In this tutorial, we demonstrated a new way to custom build PyTorch's efficient mobile interpreter - tracing-based selective build, in an Android and iOS app.

We walked through an Image Segmentation example to show how to bundle inputs to a model, generated operator list by tracing the model with bundled input, and build a custom torch library from source with the operator list from tracing result.

The custom build is still under development, and we will continue improving its size in the future. Note, however, that the APIs are subject to change in future versions.

Thanks for reading! As always, we welcome any feedback, so please create an issue here <https://github.com/pytorch/pytorch/issues>`.

Learn More


- To learn more about PyTorch Mobile, please refer to PyTorch Mobile Home Page <https://pytorch.org/mobile/home/>

* To learn more about Image Segmentation, please refer to the Image Segmentation DeepLabV3 on Android Recipe <https://tutorials.pytorch.kr/beginner/deeplabv3_on_android.html>_
