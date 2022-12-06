'''
https://wikidocs.net/52460

typical 2d tensor

|t| = (batch_size, dim)

    -----------------
    |               |
    |               |
batch_size          |
    |               |
    |               |
    -------dim-------

3d tensor for vision

|t| = (batch_size, width, height)

         ---------------
       /               /
    height            / |
     /               /  |
    -----------------   | 
    |               |   |
    |               |   |
batch_size          |   /
    |               |  /
    |               | /
    ------width------


3d tensor for nlp

|t| = (batch_size, length, dim)

         ---------------
       /               /
    dim               / |
     /               /  |
    -----------------   | 
    |               |   |
    |               |   |
batch_size          |   /
    |               |  /
    |               | /
    ------length-----

1. 
[[나는 사과를 좋아해], 
[나는 바나나를 좋아해], 
[나는 사과를 싫어해], 
[나는 바나나를 싫어해]]
2. 
[['나는', '사과를', '좋아해'], 
['나는', '바나나를', '좋아해'], 
['나는', '사과를', '싫어해'], 
['나는', '바나나를', '싫어해']]
3. 
'나는' = [0.1, 0.2, 0.9]
'사과를' = [0.3, 0.5, 0.1]
'바나나를' = [0.3, 0.5, 0.2]
'좋아해' = [0.7, 0.6, 0.5]
'싫어해' = [0.5, 0.6, 0.7]
4. -> 4x3x3
[[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.7, 0.6, 0.5]],
 [[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.7, 0.6, 0.5]],
 [[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.5, 0.6, 0.7]],
 [[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.5, 0.6, 0.7]]]
5. batch_size -> 2로 변경 2x3x3
첫번째 배치 #1
[[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.7, 0.6, 0.5]],
 [[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.7, 0.6, 0.5]]]

두번째 배치 #2
[[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.5, 0.6, 0.7]],
 [[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.5, 0.6, 0.7]]]

'''

import torch
import numpy as np


def print_func(func, *args, **kwargs):
    def f(*args, **kwargs):
        print('-'*50)
        print(f'Function: {func.__name__}() begin')
        func(*args, **kwargs)
        print(f'Function: {func.__name__}() end')

    return f


@print_func
def t_1d_manip():
    t_1d = np.array([0., 1., 2., 3., 4., 5., 6.])

    print(t_1d)
    print('Rank of t_1d: ', t_1d.ndim)
    # Rank of t_1d:  1
    print('Shape of t_1d: ', t_1d.shape)
    # Shape of t_1d:  (7,) ->(1x7)
    print('t_1d[0] t_1d[1] t_1d[-1] = ', t_1d[0], t_1d[1], t_1d[-1])
    # t_1d[0] t_1d[1] t_1d[-1] =  0.0 1.0 6.0
    print('t_1d[2:4] t_1d[4:-1] = ', t_1d[2:4], t_1d[4:-1])
    # t_1d[2:4] t_1d[4:-1] =  [2. 3.] [4. 5.]

    t_1d = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
    print(t_1d)
    print(t_1d.dim())  # rank. 즉, 차원
    print(t_1d.shape)  # shape
    print(t_1d.size())  # shape
    print('t_1d[0] t_1d[1] t_1d[-1] = ',
          t_1d[0], t_1d[1], t_1d[-1])   # 인덱스로 접근
    print('t_1d[2:4] t_1d[4:-1] = ', t_1d[2:4], t_1d[4:-1])    # 슬라이싱
    print(t_1d[:2], t_1d[3:])       # 슬라이싱


@print_func
def t_2d_manip():
    t_2d = np.array([[1., 2., 3.], [4., 5., 6.], [
                    7., 8., 9.], [10., 11., 12.]])

    print(t_2d)
    print('Rank of t_1d: ', t_2d.ndim)
    # Rank of t_1d:  2
    print('Shape of t_1d: ', t_2d.shape)
    # Shape of t_1d:  (4, 3)

    t_2d = torch.FloatTensor([[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.],
                              [10., 11., 12.]
                              ])
    print(t_2d)
    print(t_2d.dim())  # rank. 즉, 차원
    print(t_2d.size())  # shape
    print(t_2d[:, 0])
    print(t_2d[:, 1])
    print(t_2d[:, 0:2])
    print(t_2d[:, :-1])


@print_func
def tensor_add():
    m1 = torch.FloatTensor([[3., 3.]])
    m2 = torch.FloatTensor([[2., 2.]])
    print(m1.dim(), m1.size())
    print(m1+m2)


@print_func
def broad_casting():
    m1 = torch.FloatTensor([[1, 2]])
    m2 = torch.FloatTensor([3])  # [3] -> [3, 3]
    print(m1 + m2)
    # 2 x 1 Vector + 1 x 2 Vector
    m1 = torch.FloatTensor([[1, 2]])  # [[1,2],[1,2]]
    m2 = torch.FloatTensor([[3], [4]])  # [[3,3,[4,4]]
    v1 = m1 + m2

    m1 = torch.FloatTensor([[1, 2], [1, 2]])
    m2 = torch.FloatTensor([[3, 3], [4, 4]])
    v2 = m1 + m2
    if torch.all(torch.eq(v1, v2)):
        print('Same')


broad_casting()


@print_func
def mat_mul_N_mul():
    m1 = torch.FloatTensor([[1, 2], [3, 4]])
    m2 = torch.FloatTensor([[1], [2]])
    # [[1*1 + 2*2], [3*1 + 4*2]]
    print(f'M1 Shape: {m1.shape} , M2 Shape: {m2.shape}')
    m3 = m1.matmul(m2)
    print(f'm1.matmul(m2): {m3}, Shape: {m3.shape}')
    m4 = m1.mul(m2)  # m2 broad casting [[1,1], [2,2]]
    print(f'm1.mul(m2): {m4}, Shape: {m4.shape}')


mat_mul_N_mul()


@print_func
def mean_sum():
    t = torch.FloatTensor([[1, 2], [3, 4]])
    print(t.mean(),
          t.mean(dim=0),    # [[(1+3)/2],[(2+4)/2]]
          t.mean(dim=1),    # [[(1+2)/2],[(3+4)/2]]
          t.sum(),
          t.sum(dim=0),
          t.sum(dim=1))


mean_sum()


@print_func
def max_arg_max():
    t = torch.FloatTensor([[1, 2], [3, 4]])
    print(t.max(),
          t.max(dim=0),
          t.max(dim=1))


max_arg_max()


@print_func
def view():
    t = np.array([[[0, 1, 2],
                   [3, 4, 5]],
                  [[6, 7, 8],
                   [9, 10, 11]]])
    ft = torch.FloatTensor(t)
    print(ft, ft.size())
    print(ft.view(-1, 3), ft.view(-1, 3).size())
    print(ft.view(4, -1), ft.view(4, -1).size())
    print(ft.view(-1, 1, 3), ft.view(-1, 1, 3).size())


view()


@print_func
def squeeze_unsqueeze():
    # squeeze 차원이 1인 경우에는 해당 차원을 제거
    ft = torch.FloatTensor([[0], [1], [2]])
    print(ft.squeeze())
    # unsqueeze 특정 위치에 1인 차원을 추가
    ft = torch.Tensor([0, 1, 2])
    print(ft.unsqueeze(0),  # [[0, 1, 2]]
          ft.unsqueeze(1),  # [[0], [1], [2]]
          ft.unsqueeze(-1))


squeeze_unsqueeze()


@print_func
def cat():
    x = torch.FloatTensor([[1, 2], [3, 4]])
    y = torch.FloatTensor([[5, 6], [7, 8]])
    print(torch.cat([x, y], dim=0))
    print(torch.cat([x, y], dim=1))


cat()


@print_func
def stacking():
    x = torch.FloatTensor([1, 4])
    y = torch.FloatTensor([2, 5])
    z = torch.FloatTensor([3, 6])
    print(torch.stack([x, y, z]))
    print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
    print(torch.stack([x, y, z], dim=1))


stacking()


@print_func
def in_place_op():
    x = torch.FloatTensor([[1, 2], [3, 4]])
    print(x.mul_(2.))
    print(x)


in_place_op()
