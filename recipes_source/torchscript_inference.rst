TorchScript를 통해 배포하기
==========================

이번 레시피를 통해 아래와 같은 것을 배울 수 있습니다.

- TorchsScript의 개념
- TorchScript 형식에 맞게 학습된 모델을 내보내는 법
- C++에서 TorchScript 모델을 불러와 추론(inference)하는 법

요구사항(Requirements)
------------

-  PyTorch 1.5
-  TorchVision 0.6.0
-  libtorch 1.5
-  C++ compiler

위의 세 가지 PyTorch components들을 설치하는 방법은 `pytorch.org`_에 나와있습니다. 
C++ 컴파일러는 사용자의 플랫폼에 따라 달라집니다.

TorchScript 소개
--------------------

**TorchScript** 는 C++ 같은 고성능 환경에서 실행할 수 있는 PyTorch 모델(``nn.Module``의 하위 클래스)의 중간 표현(intermediate representation)입니다.
이는 모델 연산의 런타임 최적화를 실행하는 **PyTorch JIT Compiler** 에 의해 사용되는 Python의 고성능 subset입니다.
TorchScript는 Pytorch 모델을 scalable한 추론을 위해 권장되는 모델 형식입니다.
추가 정보를 위해서 `pytorch.org`_의 `TorchScript 튜토리얼 시작하기`_, `TorchScript 모델을 C++에서 불러오는 튜토리얼`_, `TorchScript 문서`_를 참고하세요.

모델 내보내기
------------------------

예시로 사전학습된 비전 모델을 가져와 보겠습니다. 모든 사전학습된 TorchVision 모델들은 TorchScript에서 사용할 수 있습니다.

아래의 Python3 코드를 스크립트나 REPL로 실행해보세요.

.. code:: python3

   import torch
   import torch.nn.functional as F
   import torchvision.models as models

   r18 = models.resnet18(pretrained=True)       # We now have an instance of the pretrained model
   r18_scripted = torch.jit.script(r18)         # *** This is the TorchScript export
   dummy_input = torch.rand(1, 3, 224, 224)     # We should run a quick test

빠진 것이 있는지 두 모델의 동등성을 검사하겠습니다.:

::

   unscripted_output = r18(dummy_input)         # Get the unscripted model's prediction...
   scripted_output = r18_scripted(dummy_input)  # ...and do the same for the scripted version

   unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
   scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

   print('Python model top 5 results:\n  {}'.format(unscripted_top5))
   print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))

두 가지의 버전의 모델 모두 같은 결과가 나오는 것을 볼 수 있습니다.:

::

   Python model top 5 results:
     tensor([[463, 600, 731, 899, 898]])
   TorchScript model top 5 results:
     tensor([[463, 600, 731, 899, 898]])

체크가 완료되었으니 모델을 저장하겠습니다.:

::

   r18_scripted.save('r18_scripted.pt')

C++에서 TorchScript 모델 불러오기
---------------------------------

Create the following C++ file and name it ``ts-infer.cpp``:

.. code:: cpp

   #include <torch/script.h>
   #include <torch/nn/functional/activation.h>


   int main(int argc, const char* argv[]) {
       if (argc != 2) {
           std::cerr << "usage: ts-infer <path-to-exported-model>\n";
           return -1;
       }

       std::cout << "Loading model...\n";

       // deserialize ScriptModule
       torch::jit::script::Module module;
       try {
           module = torch::jit::load(argv[1]);
       } catch (const c10::Error& e) {
           std::cerr << "Error loading model\n";
           std::cerr << e.msg_without_backtrace();
           return -1;
       }

       std::cout << "Model loaded successfully\n";

       torch::NoGradGuard no_grad; // ensures that autograd is off
       module.eval(); // turn off dropout and other training-time layers/functions

       // create an input "image"
       std::vector<torch::jit::IValue> inputs;
       inputs.push_back(torch::rand({1, 3, 224, 224}));

       // execute model and package output as tensor
       at::Tensor output = module.forward(inputs).toTensor();

       namespace F = torch::nn::functional;
       at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
       std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
       at::Tensor top5 = std::get<1>(top5_tensor);

       std::cout << top5[0] << "\n";

       std::cout << "\nDONE\n";
       return 0;
   }

This program:

-  Loads the model you specify on the command line
- Creates a dummy “image” input tensor
- Performs inference on the input

Also, notice that there is no dependency on TorchVision in this code.
The saved version of your TorchScript model has your learning weights
*and* your computation graph - nothing else is needed.

Building and Running Your C++ Inference Engine
----------------------------------------------

Create the following ``CMakeLists.txt`` file:

::

   cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
   project(custom_ops)

   find_package(Torch REQUIRED)

   add_executable(ts-infer ts-infer.cpp)
   target_link_libraries(ts-infer "${TORCH_LIBRARIES}")
   set_property(TARGET ts-infer PROPERTY CXX_STANDARD 11)

Make the program:

::

   cmake -DCMAKE_PREFIX_PATH=<path to your libtorch installation>
   make

Now, we can run inference in C++, and verify that we get a result:

::

   $ ./ts-infer r18_scripted.pt
   Loading model...
   Model loaded successfully
    418
    845
    111
    892
    644
   [ CPULongType{5} ]

   DONE

Important Resources
-------------------

-  `pytorch.org`_ for installation instructions, and more documentation
   and tutorials.
-  `Introduction to TorchScript tutorial`_ for a deeper initial
   exposition of TorchScript
-  `Full TorchScript documentation`_ for complete TorchScript language
   and API reference

.. _pytorch.org: https://pytorch.org/
.. _Introduction to TorchScript tutorial: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
.. _Full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
.. _Loading A TorchScript Model in C++ tutorial: https://pytorch.org/tutorials/advanced/cpp_export.html
.. _full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
