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
np.split(arr,5)
