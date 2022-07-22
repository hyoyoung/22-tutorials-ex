모듈 매개변수 초기화 건너뛰기
===========================

소개
----

모듈이 생성될 때, 모듈 유형과 관련된 기본 초기화 스킴에 따라 학습 가능한 매개변수가 초기화됩니다.
예를 들어, :class:`torch.nn.Linear` 모듈의 `weight` 매개변수는 
`uniform(-1/sqrt(in_features), 1/sqrt(in_features))` 분포로 초기화 됩니다.
기존에는 다른 초기화 스킴이 필요한 경우 모듈 인스턴스화 후 매개변수를 재초기화해야 했습니다.

::

    from torch import nn

    # 기본 분포로 가중치를 초기화합니다: uniform(-1/sqrt(10), 1/sqrt(10)).
    m = nn.Linear(10, 5)

    # 다른 분포로 가중치를 재초기화합니다.
    nn.init.orthogonal_(m.weight)

이 경우 구성 중 수행되는 초기화는 계산 낭비이며, `weight` 매개변수가 크면 사소한 문제가 아닐 수 있습니다.

초기화 건너뛰기
--------------

모듈 구성 중 매개변수 초기화를 건너뛰게 되어 낭비되는 계산을 피할 수 있습니다.
:func:`torch.nn.utils.skip_init` 함수를 사용하여 쉽게 실행할 수 있습니다.
::

    from torch import nn
    from torch.nn.utils import skip_init

    m = skip_init(nn.Linear, 10, 5)

    # 예제 : 기본 이외의 매개변수 초기화를 커스텀하여 실행합니다.
    nn.init.orthogonal_(m.weight)

아래 :ref:`Updating` 섹션에 설명된 조건을 충족하는 모듈에 적용할 수 있습니다.
`torch.nn` 에 있는 모든 모듈은 조건을 충족하기 때문에 초기화 스킵을 지원하고 있습니다.

.. _Updating:

초기화 건너뛰기를 위한 모듈 업데이트
---------------------------------

:func:`torch.nn.utils.skip_init` 의 구현(참고 :ref:`Details`) 방법에 따라,
모듈이 함수와 호환되기 위한 두 가지 요구사항이 있습니다.
다음의 요구사항을 이행하면 커스텀 모듈의 매개변수 초기화 스킵 기능을 선택할 수 있습니다:

  1. 모듈을 생성할 때 매개변수와 버퍼로 전달되는 모듈의 생성자 내 `device` kwarg를 
  사용해야 합니다. 

  2. 모듈은 초기화를 제외하고 모듈의 생성자 내 매개변수 또는 버퍼 계산을 수행하지 않아야 합니다
  (i.e. `torch.nn.init`의 함수).

다음 예시는 작성된 매개변수, 버퍼 또는 서브모듈에 `device` kwarg를 전달하여 지원하도록 
모듈이 업데이트 된 것을 나타냅니다:

::

    import torch
    from torch import nn

    class MyModule(torch.nn.Module):
      def __init__(self, foo, bar, device=None):
        super().__init__()

        # ==== 사례 1: 모듈 매개변수를 직접 생성합니다. ====
        # 생성한 매개변수에 device를 전달합니다.
        self.param1 = nn.Parameter(torch.empty((foo, bar), device=device))
        self.register_parameter('param2', nn.Parameter(torch.empty(bar, device=device)))

        # meta device 지원을 확실히 하기 위해 모듈의 생성자 내 매개변수에
        # torch.nn.init의 ops 외에는 사용하지 마십시오.
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.param1)
            nn.init.uniform_(self.param2)


        # ==== 사례 2: 모듈의 서브모듈을 생성합니다. ====
        # 모든 서브 모듈은 device를 재귀적으로 전달해야 합니다.
        # torch.nn 모듈 예시입니다.
        self.fc = nn.Linear(bar, 5, device=device)

        # 컨테이너에도 동일하게 적용합니다.
        self.linears = nn.Sequential(
            nn.Linear(5, 5, device=device),
            nn.Linear(5, 1, device=device)
        )


        # ==== 사례 3: 모듈의 버퍼를 생성합니다. ====
        # 버퍼 tensor 생성 동안 device를 전달합니다.
        self.register_buffer('some_buffer', torch.ones(7, device=device))

    ...

.. _Details:

구현 세부사항
------------

내부적으로 :func:`torch.nn.utils.skip_init` 함수는 2단계 패턴으로 구현됩니다.

::

    # 1. meta device 에서 모듈을 초기화합니다; 모든 torch.nn.init ops는 
    # meta device에서 no-op 동작을 합니다.
    m = nn.Linear(10, 5, device='meta')

    # 2. 초기화되지 않은(빈) 형태의 모듈을 CPU device에 구현합니다.
    # 결과는 초기화 되지 않은 매개 변수를 가진 모듈 인스턴스입니다.
    m.to_empty(device='cpu')

모듈은 "meta" device로 인스턴스화하여 동작합니다. tensor shape 정보를 가지고 있지만 스토리지는 할당하지 않습니다.
`torch.nn.init` ops는 meta device를 위해 특별히 구현되어 있고 no-op 동작을 합니다.
이로 인해 매개변수 초기화 로직에서 본질적으로 건너뛰게 됩니다.

:ref:`Updating` 에 설명된 대로 이 패턴은 모듈 구성 중 `device` kwarg를 적절히 지원하는 모듈에서만 작동 합니다.