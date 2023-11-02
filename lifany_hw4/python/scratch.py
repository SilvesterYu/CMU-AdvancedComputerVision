import numpy as np

# mask for row-wise maximum
a = np.array([[0, 1],
             [2, 3],
            [4, 5],
            [0, 0],
        [6, 7],
          [9, 8],
          [0, 0]])

b = (a == a.max(axis=1)[:,None]).astype(int)
print(b)

# number of zero rows
zerorows = np.sum(~a.any(1))
print(zerorows)

print("-"*10)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
np.split(arr,3)

X = np.array([[1, 2],
[3, 4],
[5, 6],
[7, 8]])
W = np.array([[0.1, 0.2, 0.3],
[1.1, 1.2, 1.3]])
b = np.array([5, 6, 7])
pre_act = np.matmul(X, W) + b
print("XW", np.matmul(X, W))
print("pre", pre_act)

print("+++++")
test = np.array([[1, 2, 3, 4],
[5, 6, 7, 8],
[9, 10, 11, 12]])
thisshape = test.shape
test = test.flatten()

print(test)
print(len(test.shape))
test = test.reshape(thisshape)
print(test)
print(len(test.shape))

