#!/usr/bin/env python

# https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/beginner/blitz/autograd_tutorial.html

import torch
from torch.autograd import Variable
# autograd는 자동 미분
# 실행-기반-정의 (define by run)
# 역전파 기능이 제공됨
# autograd.Variable로 tensor를 감싸서 연산을 할 수 있음
# variable에 대해 backward()를 호출하여 자동으로 변화도(gradient)를 계산해냄(역전파)

x = Variable(torch.ones(2, 2), requires_grad=True)
print("x=", x)

y = x + 2
print("y=x+2=", y)
print(y.grad_fn) # 연산의 결과는 grad_fn을 가지게 됨

z = y * y * 3
print("z=", z)
out = z.mean()
print("out=z.mean()=", out)

# backprop
out.backward()
print("x.grad=", x.grad) # 변화도

x = torch.randn(3)
y = Variable(x, requires_grad=True)
print("x=", x)
print("y=", y)
y = x * 2
print("y=x*2=", y)
while y.data.norm() < 1000:
    y = y * 2
print(y)

#gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
#print("gradients=", gradients)
#y.backward(gradients)
#print("x.grad=", x.grad)

