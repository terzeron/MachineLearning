#!/usr/bin/env python

import numpy as np

print("np.version.full_version=", np.version.full_version)

a = np.array([0, 1, 2, 3, 4, 5])
print("a=", a)
print("a.ndim=", a.ndim) # 차원
print("a.shape=", a.shape) # 형태

b = a.reshape((3, 2))
print("b=", b)
print("b.ndim=", b.ndim)
print("b.shape=", b.shape)

print("=" * 80)

b[1][0] = 77
print("b=", b) # b의 일부 요소 값이 변경됨
print("a=", a) # a도 변경됨

c = a.reshape((3, 2)).copy()
print("c=", c)
c[0][0] = -99
print("c=", c) # c의 일부 요소 값이 변경됨
print("a=", a) # copy했으므로 a는 변경되지 않음

print("=" * 80)

print("[1,2,3,4,5] * 2=", [1, 2, 3, 4, 5] * 2) # 길이가 확장됨
print("np.array([1,2,3,4,5]) * 2=", np.array([1, 2, 3, 4, 5]) * 2) # 배열 내의 개별 요소가 2배 

c = np.array([1, 2, np.nan, 3, 4])
print("c=", c)
print("np.isnan(c)=", np.isnan(c))
print("c[~np.isnan(c)]=", c[~np.isnan(c)]) # 조건에 맞는 요소만 추려냄
print("np.mean(c[~np.isnan(c)])=", np.mean(c[~np.isnan(c)])) 




