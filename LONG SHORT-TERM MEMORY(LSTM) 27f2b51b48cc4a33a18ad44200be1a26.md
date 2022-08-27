# LONG SHORT-TERM MEMORY(LSTM)

## 0. Abstract

Learning to store information over extended time intervals via recurrent backpropagation
takes a very long time, mostly due to insufficient, decaying error back flow. We briefly review
Hochreiter's 1991 analysis of this problem, then address it by introducing a novel, efficient,
gradient-based method called “Long Short-Term Memory" (LSTM). Truncating the gradient
where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000
discrete time steps by enforcing constant error flow through “constant error carrousels" within
special units. Multiplicative gate units learn to open and close access to the constant error
flow. LSTM is local in space and time; its computational complexity per time step and weight
is O(1). Our experiments with artificial data involve local, distributed, real-valued, and noisy
pattern representations. In comparisons with RTRL, BPTT, Recurrent Cascade-Correlation,
Elman nets, and Neural Sequence Chunking, LSTM leads to many more successful runs, and
learns much faster. LSTM also solves complex, articial long time lag tasks that have never
been solved by previous recurrent network algorithms.

---

기존RNN은 시간이 오래걸리고 대부분 기울기가 소실되는 문제가 발생했다.이러한 한계점은 Error back flow( back propagation)과정이 정보를 충분히 전달하지 못하기 때문이고, 수 많은 layer를 지나면서 Weight가 vanishing 되기 때문이다. 본 논문에서는 이러한 문제점을 해결할 수 있는 novel, efficient gradient- based method인 LSTM(Long Short-Term Memory)를 제안한다.

LSTM에서는 특정 정보가 Gradient에 안좋은 영향을 미치지 않는 한, 약 1000 번의 time step 이상의 interval에도 정보를 소실하지 않고 효과적으로 정보를 전달할 수 있다.

본 논문에서는 인위적으로 만들어낸 다양한 패턴들에 대해서 LSTM을 적용시켜 RTRL,BPTT, Recurrent Cascade-Correlation, Elman nets, Neural Sequence Chnking등과 비교해보았으며, 실험을 통해 LSTM의 우수함을 입증하였다. 단순히 성능 지표만 높을 뿐 아니라, 기존 RNN류 모델들이 풀지 못했던 Long Time Lag Task에서 최초로 성공을 거두었다고한다.

## 1. Recurrent Neural Network

1980년대에 처음 제안된 개념으로 기존에 있는 인공신경망은 해결할 수 없던 문제를 ANN으로 시계열 문제해결 어려움을 해결했다. 내부의 메모리를 이용해 시퀸스 형태의 입력처리가 가능해진다.

필기 인식, 음성 인식 등 시퀸스 데이터 처리에 적용할 수 있다.

## 1. LSTM(Long Short Term Memory) Background

**Recurrent Nenural Network(RNN) 컨셉**

![Untitled](LONG%20SHORT-TERM%20MEMORY(LSTM)%2027f2b51b48cc4a33a18ad44200be1a26/Untitled.png)

- Recurrent란, 이전에서 어떤 정보(데이터)가 추가적으로 오는 것을 말한다.
- RNN은 시간적으로 상관관계가 있는 데이터에서 주로 사용된다.
- 직전 데이터(t-1)과 현재 데이터(t) 간의 상관관계(correlation)을 고려하여 다음의 데이터(t+1)를 예측하고자, 과거의 데이터도 반영한 신경망 모델을 만든다.
- 시간을 많이 거슬러 올라갈수록(long term) 경사를 소실하는 문제가 있다.
    - 선형 함수가 아닌 비선형 함수를 활성함수로 쓰는 것과 비슷한 이유로, 초깃값에 따라서 과거 데이터를 계속 곱할수록 작아지는 문제가 발생하게 된다.
    - LSTM은 구조를 개선하여 이 문제를 해결했다.
    - [Le, Q. V., Jaitly, N., & Hinton, G. E. (2015](https://arxiv.org/abs/1504.00941)) 연구에 따르면, 활성함수를 ReLU로 사용하고 가중치를 단위행렬로 초기화하면 long-term을 학습시킬 수 있다.
- Vanilla RNN: RNN의 대표적인 모델로, 이전의 정보(**xt-1**)와 현재 정보(**xt**)를 취합(**tanh**, 하이퍼볼릭탄젠트)한 정보를 신경망에 들어가서 아웃풋(**ht**)을 만듦
- 장기 의존성 (Long-Term Dependency) 문제가 발생한다.
- RNN 처럼 직전 정보만 참고하는 것이 아니라, 그 전 정보를 고려해야 하는 경우(longer-term) 가 있음(예: 책을 읽을 때, 몇 페이지/챕터 전에 있는 정보를 머리 속에 기억하고 있어야 하는 경우
    
    I grew up in France... I speak fluent French. 문장에서 'french'를 예측하는 경우)
    
- 시퀀스가 있는 문장에서 문장 간의 간격(gap, 입력 위치의 차이)이 커질 수록, RNN은 두 정보의 맥락을 파악하기 어려워진다.
- 따라서, 한참 전의 데이터도 함께 고려하여 출력을 만들어보는 것이  LSTM의 목적이다.

## 2. LSTM(Long Short Term Memory) Definition

LSTM 컨셉은 [Hochreiter, S., & Schmidhuber, J. (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)이 제안했으며, 많은 개선을 통해 언어, 음성인식 등 다양한 분야에서 사용되고 있다.

- RNN의 주요 모델 중 하나로, 장기 의존성 문제를 해결할 수 있다.
- 직전 데이터뿐만 아니라, 좀 더 거시적으로 과거 데이터를 고려하여 미래의 데이터를 예측하기 위함이 목적이다.

바닐라 RNN보다 복잡한 구조가 왜 long-term dependency 한지 이해할수 있다.

![Untitled](LONG%20SHORT-TERM%20MEMORY(LSTM)%2027f2b51b48cc4a33a18ad44200be1a26/Untitled%201.png)

LSTM 구조

![Untitled](LONG%20SHORT-TERM%20MEMORY(LSTM)%2027f2b51b48cc4a33a18ad44200be1a26/Untitled%202.png)

- **Neural Network Layer**> 웨이트(weight)와 바이어스(bias) 둘 다 있다.
- **Pointwise Operation**> Pointwise Operation으로 연산이 되면, 각각의 차원(dimension)에 맞게 곱하거나 더하게 된다.
- **input과 output의 차원이 같다고 가정한다면,**
    
    > 100 dimension과 100 dimension이 concatenate 하게 되면 200 dimension이 되지만, Neural Network Layer는 200 dimension을 100dimension으로 가는 네트워크가 되는 걸 유추해볼 수 있다.(실제로도 input과 output의 차원이 같다.)
    

## 2-1. LSTM(Long Short Term Memory) Structure

![Untitled](LONG%20SHORT-TERM%20MEMORY(LSTM)%2027f2b51b48cc4a33a18ad44200be1a26/Untitled%203.png)

총 6개의 파라미터가 있으며, 네 개의 게이트(gate)로 이루어져 있

### **A. INPUT(Xt)**

### **B. (Cell) State**

- 회전목마 같은 구조로 인해 오차가 사라지지 않고, 전체 체인을 관통한다.
- (x)게이트 매케니즘: 정보를 여닫는 역할을 한다.

### C**. Hidden State**

: 이전 출력(previous output)

### D**. Gates (Forget GAte, Input Gate, Output Gate)**

세 개의 게이트는 정보들이 어느 시점에서 정보를 버리거나 유지하여 선택적으로 흘러갈 수 있게(=long term과 short term을 잘 고려하는) 하기 위함이다.

- 입력웨이트 충돌(input wight conflict)과 출력 웨이트 충돌(output weight conflict)
    
    ---
    
    자신이 발화해야할 신호가 전파돼 왔을 때는 웨이트를 크게 해서 활성화해야 하지만, 관계가 없는 신호가 전파되었을 때는 웨이트를 작게 해서 비활성인 채로 있어야 한다.
    
    시계열 데이터를 입력에서 받을 경우와 비교해보면, 이것은 시간 의존성이 있는 신호를 받았을 때는 웨이트를 크게 하고, 의존성이 없는 신호를 받았을 때는 웨이트를 작게 하는 것이다. 그러나 뉴런이 동일한 웨이트로 연결돼 있다면 두 가지 경우에 서로 상쇄하는 형태의 웨이트 변경이 이뤄지므로 특히 장기의존성 학습이 잘 실행되지 않게 된다.
    

**Step 1. Forget Gate: 과거 정보를 버릴지 말지 결정하는 과정**

![Untitled](LONG%20SHORT-TERM%20MEMORY(LSTM)%2027f2b51b48cc4a33a18ad44200be1a26/Untitled%204.png)

과거의 정보를 통해 맥락을 고려하는 것도 중요하지만, 그 정보가 필요하지 않을 경우에는 과감히 버리는 것도 중요하다.

- 이전 output과 현재 input을 넣어, cell state로 가는 과거 정보값이 나온다.
- 활성함수는 시그모이드(sigmoid)를 사용하므로, 0 또는 1 값이 나온다.
    
    → ‘0’일 경우, 이전의 cell state값은 모두 ‘0’이 되어 미래의 결과에 아무런 영향을 주지 않는다.
    
    → ‘1’일 경우, 미래의 예측 결과에 영향을 주도록 이전의 cell state(CT-1)을 그대로 보내 완전히 유지한다.
    
- 즉, Forget Gate는 현재 입력과 이전 출력을 고려해서, cell state의 어떤 값을 버릴지/ 지워버릴지 결정하는 역할이다.

**Step 2. Forget Gate: 과거 정보를 버릴지 말지 결정하는 과정**

![Untitled](LONG%20SHORT-TERM%20MEMORY(LSTM)%2027f2b51b48cc4a33a18ad44200be1a26/Untitled%205.png)

- 현재의 cell state 값에 얼마나 더할지 말지를 정하는 역할( tanh는 -1 에서 1사이의 값이 나온다)
- Forget Gate와 Input Gate의 주요 역할
    
    : 이전 cell state 값을 얼마나 버릴지, 지금 입력과  이전 출력으로 얻어진 값을 얼마나 cell state에 반영할지 정하는 역할이다.
    

**Step 3. Update (cell state): 과거 cell state(Ct-1)를 새로운 state(Ct)로 업데이트 하는 과정**

![Untitled](LONG%20SHORT-TERM%20MEMORY(LSTM)%2027f2b51b48cc4a33a18ad44200be1a26/Untitled%206.png)

- Forget Gate를 통해서 얼마나 버릴지, Input Gate에서 얼마나 더할지를 정했으므로,
    
    >input gate*current state+ forget*previous state
    

**Step 4. Output Gate (hidden state): 어떤 출력값을 출력할지 결정하는 과정**

![Untitled](LONG%20SHORT-TERM%20MEMORY(LSTM)%2027f2b51b48cc4a33a18ad44200be1a26/Untitled%207.png)

- Output based on the updated state.
- 최종적으로 얻어진 cell state 값을 얼마나 빼낼지 결정하는 역할이다.
    - output gate* updated state

**Step 5. Output (ht)**

• output state 는 다음 hidden state와 항상 동일함

## 3. LSTM(Long Short Term Memory) drawback

Output 게이트가 C(t)를 전달하기 때문에, LSTM 블록별 cell state는 output 게이트에 따라 달라진다.(input, forget 게이트는 C(t-1)를 전달한다)

Output 게이트가 계속 닫혀있는 경우(시그모에드에서 0을 보내는 경우를 의미한다) cell state에 접근할 수 없다는 문제가 발생하게 된다. 이 문제를 해결하기 위해 도입된 것이 ‘핍홉연결' 이다.

## 3.1 peephole connection

핍홉 연결(peephole connection) LSTM의 변종이다. 기존의 LSTM에서 gate controller($f_t,i_t,o_t$)는 입력 $x_t$와 이전 타입스텝의 단기 상태 $h_{t-1}$만 입력으로 받는다. 하지만 위의 논문에서 제안한 핍홉 연결을 아래의 그림과 같이 연결 해주면서 gate controller에 이전 타임스템의 장기 상태 $c_{t-1}$가 입력으로 추가되며, 좀 더 많은 맥락(context)를 인식할 수 있다.

![Untitled](LONG%20SHORT-TERM%20MEMORY(LSTM)%2027f2b51b48cc4a33a18ad44200be1a26/Untitled%208.png)

## 4. Conclusion

- 기존 Vanilla RNN이 가지는 문제점인 Long-term dependencies를 효과적으로 해결할 수 있는 LSTM 방식을 제안한다.
- LSTM은 Cell State 개념을 도입하여 과거의 데이터를 유지하면서도 불필요한 데이터는 Forget Gate를 통해 삭제하여 Gradient Update에 최적의 상태를 유지한다.
- 기존 RNN 방식 대비 정확도를 크게 향상시킬 수 있었으며, Long Time Lack Task의 경우 기존 RNN 모델들이 해결하지 못했던 문제를 해결할 수 있었다.