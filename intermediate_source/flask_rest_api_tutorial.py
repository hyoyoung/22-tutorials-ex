# -*- coding: utf-8 -*-
"""
Flask와 REST API를 이용한 파이토치 배포
========================================================
**Author**: `Avinash Sajjanshetty <https://avi.im>`_
**번역**: 유연승

이 튜토리얼에서는 Flask를 사용하여 파이토치 모델을 배포하고 모델의 추론결과룰 REST API로 보여줍니다.
이미지를 디텍션하는 사전학습된 DenseNet 121 모델을 배포해봅시다.

.. tip:: 여기서 사용한 모든 코드는 MIT 라이선스로 공개되고 `Github <https://github.com/avinassh/pytorch-flask-api>`_ 에서 확인하실 수 있습니다.

이는 PyTorch 모델을 상용화로 배포하는 튜토리얼 중 첫 번째입니다.
이러한 배포방법으로 플라스크를 사용하는 것이 PyTorch 모델의 서비스를 시작하는 가장 쉬운 방법이지만 고성능 요구 사항이 있는 사용 사례에서는 작동하지 않습니다.
다음을 참고하세요.

    - TorchScript에 대해 이미 잘 알고 계시다면 바로 `Loading a TorchScript Model in C++ <https://pytorch.org/tutorials/advanced/cpp_export.html>`_를 들어가보세요.

    - TorchScript에 대한 교육이 먼저 필요하다면 `Intro a TorchScript <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_부터 보세요.

"""


######################################################################
# API 정의
# --------------
#
# 먼저 API 앤드포인트와 요청 및 응답 유형을 정의할 것입니다.
# 우리의 API 앤드포인트는 이미지를 포함하는 "file" 매개 변수를 HTTP POST로 "/predict"에 요청할 것입니다.
# 이에대한 응답은 아래와 같은 예측을 포함한 JSON으로 응답할 것입니다
#
# ::
#
#     {"class_id": "n02124075", "class_name": "Egyptian_cat"}
#
#

######################################################################
# 의존성
# ------------
#
# 아래의 명령어를 실행하여 필요한 라이브러리를 설치하세요:
#
# ::
#
#     $ pip install Flask==2.0.1 torchvision==0.10.0


######################################################################
# Simple Web Server
# -----------------
#
# Following is a simple webserver, taken from Flask's documentaion


from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'

###############################################################################
# Save the above snippet in a file called ``app.py`` and you can now run a
# Flask development server by typing:
#
# ::
#
#     $ FLASK_ENV=development FLASK_APP=app.py flask run

###############################################################################
# When you visit ``http://localhost:5000/`` in your web browser, you will be
# greeted with ``Hello World!`` text

###############################################################################
# We will make slight changes to the above snippet, so that it suits our API
# definition. First, we will rename the method to ``predict``. We will update
# the endpoint path to ``/predict``. Since the image files will be sent via
# HTTP POST requests, we will update it so that it also accepts only POST
# requests:


@app.route('/predict', methods=['POST'])
def predict():
    return 'Hello World!'

###############################################################################
# We will also change the response type, so that it returns a JSON response
# containing ImageNet class id and name. The updated ``app.py`` file will
# be now:

from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})


######################################################################
# Inference
# -----------------
#
# In the next sections we will focus on writing the inference code. This will
# involve two parts, one where we prepare the image so that it can be fed
# to DenseNet and next, we will write the code to get the actual prediction
# from the model.
#
# Preparing the image
# ~~~~~~~~~~~~~~~~~~~
#
# DenseNet model requires the image to be of 3 channel RGB image of size
# 224 x 224. We will also normalise the image tensor with the required mean
# and standard deviation values. You can read more about it
# `here <https://pytorch.org/vision/stable/models.html>`_.
#
# We will use ``transforms`` from ``torchvision`` library and build a
# transform pipeline, which transforms our images as required. You
# can read more about transforms `here <https://pytorch.org/vision/stable/transforms.html>`_.

import io

import torchvision.transforms as transforms
from PIL import Image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


######################################################################
# The above method takes image data in bytes, applies the series of transforms
# and returns a tensor. To test the above method, read an image file in
# bytes mode (first replacing `../_static/img/sample_file.jpeg` with the actual
# path to the file on your computer) and see if you get a tensor back:

with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)

######################################################################
# Prediction
# ~~~~~~~~~~~~~~~~~~~
#
# Now will use a pretrained DenseNet 121 model to predict the image class. We
# will use one from ``torchvision`` library, load the model and get an
# inference. While we'll be using a pretrained model in this example, you can
# use this same approach for your own models. See more about loading your
# models in this :doc:`tutorial </beginner/saving_loading_models>`.

from torchvision import models

# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat


######################################################################
# The tensor ``y_hat`` will contain the index of the predicted class id.
# However, we need a human readable class name. For that we need a class id
# to name mapping. Download
# `this file <https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json>`_
# as ``imagenet_class_index.json`` and remember where you saved it (or, if you
# are following the exact steps in this tutorial, save it in
# `tutorials/_static`). This file contains the mapping of ImageNet class id to
# ImageNet class name. We will load this JSON file and get the class name of
# the predicted index.

import json

imagenet_class_index = json.load(open('../_static/imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


######################################################################
# Before using ``imagenet_class_index`` dictionary, first we will convert
# tensor value to a string value, since the keys in the
# ``imagenet_class_index`` dictionary are strings.
# We will test our above method:


with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))

######################################################################
# You should get a response like this:

['n02124075', 'Egyptian_cat']

######################################################################
# The first item in array is ImageNet class id and second item is the human
# readable name.
#
# .. Note ::
#    Did you notice that ``model`` variable is not part of ``get_prediction``
#    method? Or why is model a global variable? Loading a model can be an
#    expensive operation in terms of memory and compute. If we loaded the model in the
#    ``get_prediction`` method, then it would get unnecessarily loaded every
#    time the method is called. Since, we are building a web server, there
#    could be thousands of requests per second, we should not waste time
#    redundantly loading the model for every inference. So, we keep the model
#    loaded in memory just once. In
#    production systems, it's necessary to be efficient about your use of
#    compute to be able to serve requests at scale, so you should generally
#    load your model before serving requests.

######################################################################
# Integrating the model in our API Server
# ---------------------------------------
#
# In this final part we will add our model to our Flask API server. Since
# our API server is supposed to take an image file, we will update our ``predict``
# method to read files from the requests:
#
# .. code-block:: python
#
#    from flask import request
#
#    @app.route('/predict', methods=['POST'])
#    def predict():
#        if request.method == 'POST':
#            # we will get the file from the request
#            file = request.files['file']
#            # convert that to bytes
#            img_bytes = file.read()
#            class_id, class_name = get_prediction(image_bytes=img_bytes)
#            return jsonify({'class_id': class_id, 'class_name': class_name})

######################################################################
# The ``app.py`` file is now complete. Following is the full version; replace
# the paths with the paths where you saved your files and it should run:
#
# .. code-block:: python
#
#   import io
#   import json
#
#   from torchvision import models
#   import torchvision.transforms as transforms
#   from PIL import Image
#   from flask import Flask, jsonify, request
#
#
#   app = Flask(__name__)
#   imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
#   model = models.densenet121(pretrained=True)
#   model.eval()
#
#
#   def transform_image(image_bytes):
#       my_transforms = transforms.Compose([transforms.Resize(255),
#                                           transforms.CenterCrop(224),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize(
#                                               [0.485, 0.456, 0.406],
#                                               [0.229, 0.224, 0.225])])
#       image = Image.open(io.BytesIO(image_bytes))
#       return my_transforms(image).unsqueeze(0)
#
#
#   def get_prediction(image_bytes):
#       tensor = transform_image(image_bytes=image_bytes)
#       outputs = model.forward(tensor)
#       _, y_hat = outputs.max(1)
#       predicted_idx = str(y_hat.item())
#       return imagenet_class_index[predicted_idx]
#
#
#   @app.route('/predict', methods=['POST'])
#   def predict():
#       if request.method == 'POST':
#           file = request.files['file']
#           img_bytes = file.read()
#           class_id, class_name = get_prediction(image_bytes=img_bytes)
#           return jsonify({'class_id': class_id, 'class_name': class_name})
#
#
#   if __name__ == '__main__':
#       app.run()

######################################################################
# Let's test our web server! Run:
#
# ::
#
#     $ FLASK_ENV=development FLASK_APP=app.py flask run

#######################################################################
# We can use the
# `requests <https://pypi.org/project/requests/>`_
# library to send a POST request to our app:
#
# .. code-block:: python
#
#   import requests
#
#   resp = requests.post("http://localhost:5000/predict",
#                        files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})

#######################################################################
# Printing `resp.json()` will now show the following:
#
# ::
#
#     {"class_id": "n02124075", "class_name": "Egyptian_cat"}
#

######################################################################
# Next steps
# --------------
#
# The server we wrote is quite trivial and and may not do everything
# you need for your production application. So, here are some things you
# can do to make it better:
#
# - The endpoint ``/predict`` assumes that always there will be a image file
#   in the request. This may not hold true for all requests. Our user may
#   send image with a different parameter or send no images at all.
#
# - The user may send non-image type files too. Since we are not handling
#   errors, this will break our server. Adding an explicit error handing
#   path that will throw an exception would allow us to better handle
#   the bad inputs
#
# - Even though the model can recognize a large number of classes of images,
#   it may not be able to recognize all images. Enhance the implementation
#   to handle cases when the model does not recognize anything in the image.
#
# - We run the Flask server in the development mode, which is not suitable for
#   deploying in production. You can check out `this tutorial <https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/>`_
#   for deploying a Flask server in production.
#
# - You can also add a UI by creating a page with a form which takes the image and
#   displays the prediction. Check out the `demo <https://pytorch-imagenet.herokuapp.com/>`_
#   of a similar project and its `source code <https://github.com/avinassh/pytorch-flask-api-heroku>`_.
#
# - In this tutorial, we only showed how to build a service that could return predictions for
#   a single image at a time. We could modify our service to be able to return predictions for
#   multiple images at once. In addition, the `service-streamer <https://github.com/ShannonAI/service-streamer>`_
#   library automatically queues requests to your service and samples them into mini-batches
#   that can be fed into your model. You can check out `this tutorial <https://github.com/ShannonAI/service-streamer/wiki/Vision-Recognition-Service-with-Flask-and-service-streamer>`_.
#
# - Finally, we encourage you to check out our other tutorials on deploying PyTorch models
#   linked-to at the top of the page.
