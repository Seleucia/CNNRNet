import  numpy as np
data=[
[ 0., 2., 2., 0., 2., 0., 1.],
[ 2., 1., 2., 2., 2., 0., 2.],
[ 0., 0., 2., 0., 2., 2., 2.],
[ 0., 2., 1., 0., 1., 2., 2.],
[ 0., 2., 2., 0., 0., 0., 2.],
[ 0., 2., 0., 0., 2., 0., 2.],
[ 0., 2., 2., 0., 2., 0., 1.],
[ 1., 2., 2., 1., 2., 1., 0.],
[ 0., 0., 0., 0., 0., 0., 0.],
[ 1., 2., 2., 1., 2., 1., 0.],
]
data_num = len(data)

one_hot_dat = np.zeros((data_num, 3**7))
# vector for making indices
vec = np.asarray([3**i for i in range(7)])
# compute the corresponding index for each data point
hot_idx = np.sum(np.asarray(data)*vec, axis=1).astype(int)
one_hot_dat[range(data_num), hot_idx] = 1
print(one_hot_dat)