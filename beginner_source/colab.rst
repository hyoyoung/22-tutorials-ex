UsiColab에서 Google Drive의 Tutorial Data 사용하기
==============================================

사용자가 Google Colab의 튜토리얼과 관련된 노트북을 열 수 있도록
새로운 기능이 튜토리얼에 추가되었습니다.
복잡한 tutorial들을 실행하려면 사용자의 Google drive 계정에
데이터를 복사해야할 수도 있습니다.

이번 예제에서, 챗봇 튜토리얼을 Colab에서 동작하도록 변경하는 방법을 설명하겠습니다.
이를 위해서, 먼저 Goggle Drive에 로그인해야 합니다.
(Colab의 데이터에 접근하는 방법에 대해 자세히 알고싶다면,
`여기 <https://colab.research.google.com/notebooks/io.ipynb#scrollTo=XDg9OBaYqRMd>`__
에서 예제 노트북을 통해 볼 수 있습니다.)

시작하기 전에 `챗봇
튜토리얼 <https://pytorch.org/tutorials/beginner/chatbot_tutorial.html>`__
을 브라우저에서 열어주세요.

페이지 상단에 **Run in Google Colab** 을 클릭합니다.

Colab에서 파일이 열립니다.

**Runtime** 을 선택한 뒤, **Run All** 을 선택하면,
파일을 찾을 수 없다(the file can't be found)는 에러가 발생합니다.

이를 해결하기 위해, 필요한 파일을 Google Drive 계정으로 복사하겠습니다.

1. Google Drive에 로그인합니다.
2. Google Drive에서, **data** 라는 이름으로 폴더를 생성하고, 하위 폴더로
   **cornell** 을 생성합니다.
3. Cornell Movie Dialogs Corpus에 방문하여 ZIP 파일을 다운로드하세요.
4. 로컬 머신에 압축을 풉니다.
5. **movie\_lines.txt** 와 **movie\_conversations.txt** 파일을
   Google Drive에 생성한 **data/cornell** 폴더로 복사합니다.

이제 Google Drive 안의 파일을 가리키도록\_ \_Colab 파일을 편집하겠습니다.

Colab에서, *corpus\_name* 으로 시작하는 코드 섹션의 윗 부분에
다음 내용을 추가합니다:

::

    from google.colab import drive
    drive.mount('/content/gdrive')

2줄을 다음과 같이 변경합니다:

1. **corpus\_name** 값을 **"cornell"** 로 변경합니다.
2. **corpus** 로 시작하는 줄을 다음과 같이 변경합니다:

::

    corpus = os.path.join("/content/gdrive/My Drive/data", corpus_name)

We're now pointing to the file we uploaded to Drive.

Now when you click the **Run cell** button for the code section,
you'll be prompted to authorize Google Drive and you'll get an
authorization code. Paste the code into the prompt in Colab and you
should be set.

Rerun the notebook from the **Runtime** / **Run All** menu command and
you'll see it process. (Note that this tutorial takes a long time to
run.)

Hopefully this example will give you a good starting point for running
some of the more complex tutorials in Colab. As we evolve our use of
Colab on the PyTorch tutorials site, we'll look at ways to make this
easier for users.
