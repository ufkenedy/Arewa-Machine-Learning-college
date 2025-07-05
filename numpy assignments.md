### 100 numpy exercises

### 1. Import the numpy package under the name np (★☆☆)

```python
import numpy as np
```

### 2. Print the numpy version and the configuration (★☆☆)

```python
print(np.__version__)
print(np.show_config())
```

### 3. Create a null vector of size 10 (★☆☆)

```python
np.zeros(10)
```

### 4. How to find the memory size of any array (★☆☆)

```python
a = np.zeros(10)
print(a.nbytes)
```

### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)

```bash
python -c "import numpy; help(numpy.add)"
```

Or inside Python:

```python
help(np.add)
```

### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

```python
a = np.zeros(10)
a[4] = 1
print(a)
```

### 7. Create a vector with values ranging from 10 to 49 (★☆☆)

```python
np.arange(10, 50)
```

### 8. Reverse a vector (first element becomes last) (★☆☆)

```python
a = np.arange(10)
a[::-1]
```

### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

```python
np.arange(9).reshape(3, 3)
```

### 10. Find indices of non-zero elements from  (★☆☆)

```python
np.nonzero([1, 2, 0, 0, 4, 0])
```

### 11. Create a 3x3 identity matrix (★☆☆)

```python
np.eye(3)
```

### 12. Create a 3x3x3 array with random values (★☆☆)

```python
np.random.random((3, 3, 3))
```

### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

```python
a = np.random.random((10, 10))
print(a.min(), a.max())
```

### 14. Create a random vector of size 30 and find the mean value (★☆☆)

```python
a = np.random.random(30)
print(a.mean())
```

### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

```python
a = np.ones((5, 5))
a[1:-1, 1:-1] = 0
print(a)
```

### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

```python
a = np.ones((3, 3))
np.pad(a, pad_width=1, mode='constant', constant_values=0)
```

### 17. What is the result of the following expression? (★☆☆)

```python
import numpy as np

print(0 * np.nan)               # nan
print(np.nan == np.nan)         # False
print(np.inf > np.nan)          # False
print(np.nan - np.nan)          # nan
print(np.nan in set([np.nan]))  # True
print(0.3 == 3 * 0.1)           # False (due to floating point precision)
```

### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

```python
a = np.diag(1 + np.arange(4), k=-1)
print(a)
```

### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

```python
a = np.zeros((8, 8), dtype=int)
a[1::2, ::2] = 1
a[::2, 1::2] = 1
print(a)
```

### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)

```python
print(np.unravel_index(99, (6, 7, 8)))  # (1, 5, 3)
```

### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

```python
a = np.tile([[0, 1], [1, 0]], (4, 4))
print(a)
```

### 22. Normalize a 5x5 random matrix (★☆☆)

```python
a = np.random.random((5, 5))
a = (a - a.min()) / (a.max() - a.min())
print(a)
```

### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

```python
color = np.dtype([('r', np.ubyte), ('g', np.ubyte), ('b', np.ubyte), ('a', np.ubyte)])
print(color)
```

### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

```python
a = np.random.random((5, 3))
b = np.random.random((3, 2))
print(np.dot(a, b))
```

### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

```python
a = np.arange(11)
a[(a > 3) & (a < 8)] *= -1
print(a)
```

### 26. What is the output of the following script? (★☆☆)

```python
print(sum(range(5), -1))  # 9

from numpy import *
print(sum(range(5), -1))  # 10
```

Explanation: Python's built-in sum(range(5), -1) sums 0+1+2+3+4 -1 = 9. Numpy's sum adds differently.

### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

```python
Z = np.arange(5)
print(Z**Z)      # legal
print(2 << Z >> 2)  # legal
print(Z < -Z)    # legal
print(1j*Z)      # legal
print(Z/1/1)     # legal
# print(Z < Z > Z) # illegal, syntax error
```

### 28. What are the result of the following expressions? (★☆☆)

```python
print(np.array(0) / np.array(0))    # nan with RuntimeWarning
print(np.array(0) // np.array(0))   # raises ZeroDivisionError or RuntimeWarning
print(np.array([np.nan]).astype(int).astype(float))  # array([-2.14748365e+09]) (undefined behavior)
```

### 29. How to round away from zero a float array ? (★☆☆)

```python
a = np.array([-1.5, 1.5, -0.5, 0.5])
np.copysign(np.ceil(np.abs(a)), a)
```

### 30. How to find common values between two arrays? (★☆☆)

```python
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.intersect1d(a, b)
```

### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

```python
import warnings
warnings.filterwarnings('ignore')
```

Or specifically for numpy:

```python
np.seterr(all='ignore')
```

### 32. Is the following expressions true? (★☆☆)

```python
print(np.sqrt(-1) == np.emath.sqrt(-1))  # False
```

Explanation: `np.sqrt(-1)` returns nan, while `np.emath.sqrt(-1)` returns 1j.

### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

```python
today = np.datetime64('today', 'D')
yesterday = today - np.timedelta64(1, 'D')
tomorrow = today + np.timedelta64(1, 'D')
print(yesterday, today, tomorrow)
```

### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

```python
np.arange('2016-07', '2016-08', dtype='datetime64[D]')
```

### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)

```python
A = np.ones(3)
B = np.ones(3)
A += B
A *= -A / 2
print(A)
```

Note: This modifies A in place.

### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)

```python
a = np.random.uniform(0, 10, 10)
print(a - a % 1)
print(np.floor(a))
print(a.astype(int))
print(np.trunc(a))
```

### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

```python
np.tile(np.arange(5), (5, 1))
```

### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

```python
def gen():
    for i in range(10):
        yield i

a = np.fromiter(gen(), dtype=int, count=10)
print(a)
```

### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

```python
np.linspace(0, 1, 12)[1:-1]
```

### 40. Create a random vector of size 10 and sort it (★★☆)

```python
a = np.random.random(10)
a.sort()
print(a)
```

### 41. How to sum a small array faster than np.sum? (★★☆)

```python
a = np.arange(10)
sum(a)  # Python built-in sum is faster for small arrays
```

### 42. Consider two random array A and B, check if they are equal (★★☆)

```python
A = np.random.random(10)
B = A.copy()
np.array_equal(A, B)
```

### 43. Make an array immutable (read-only) (★★☆)

```python
a = np.arange(10)
a.flags.writeable = False
```

### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

```python
import numpy as np

z = np.random.random((10, 2))
x, y = z[:, 0], z[:, 1]
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
print(r)
print(theta)
```

### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

```python
a = np.random.random(10)
a[a.argmax()] = 0
print(a)
```

### 46. Create a structured array with x and y coordinates covering the [1]x[1] area (★★☆)

```python
dtype = [('x', float), ('y', float)]
arr = np.zeros((5, 5), dtype=dtype)
arr['x'], arr['y'] = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
print(arr)
```

### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)

```python
X = np.arange(5)
Y = X + 0.5
C = 1.0 / (X[:, None] - Y[None, :])
print(C)
```

### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

```python
for dtype in [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]:
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
    print(f"{dtype}: min={info.min}, max={info.max}")
```

### 49. How to print all the values of an array? (★★☆)

```python
np.set_printoptions(threshold=np.inf)
print(np.arange(1000))
```

### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

```python
a = np.random.random(10)
v = 0.5
index = (np.abs(a - v)).argmin()
closest_value = a[index]
print(closest_value)
```

### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

```python
dtype = [('position', [('x', float), ('y', float)]), ('color', [('r', int), ('g', int), ('b', int)])]
arr = np.zeros(3, dtype=dtype)
print(arr)
```

### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

```python
points = np.random.random((100, 2))
distances = np.sqrt(np.sum((points[:, None, :] - points[None, :, :])**2, axis=-1))
print(distances)
```

### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

You cannot convert dtype in place because it changes memory layout. You must create a new array:

```python
a = np.array([1.5, 2.3, 3.7], dtype=np.float32)
a_int = a.astype(np.int32)
print(a_int)
```

### 54. How to read the following file? (★★☆)

File content:
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

```python
import numpy as np
data = np.genfromtxt('file.csv', delimiter=',', filling_values=np.nan)
print(data)
```

### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

```python
for index, value in np.ndenumerate(arr):
    print(index, value)
```

### 56. Generate a generic 2D Gaussian-like array (★★☆)

```python
x, y = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
d = np.sqrt(x*x + y*y)
sigma, mu = 0.4, 0.0
g = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))
print(g)
```

### 57. How to randomly place p elements in a 2D array? (★★☆)

```python
a = np.zeros((10,10))
p = 5
indices = np.random.choice(a.size, p, replace=False)
np.put(a, indices, 1)
print(a)
```

### 58. Subtract the mean of each row of a matrix (★★☆)

```python
a = np.random.random((3,4))
a -= a.mean(axis=1, keepdims=True)
print(a)
```

### 59. How to sort an array by the nth column? (★★☆)

```python
a = np.random.randint(0,10,(5,5))
n = 2
a_sorted = a[a[:, n].argsort()]
print(a_sorted)
```

### 60. How to tell if a given 2D array has null columns? (★★☆)

```python
a = np.array([[1,0,3],[4,0,6]])
null_columns = np.any(a != 0, axis=0) == False
print(null_columns)
```

### 61. Find the nearest value from a given value in an array (★★☆)

Same as 50.

### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

```python
a = np.arange(3).reshape(1,3)
b = np.arange(3).reshape(3,1)
c = np.add.outer(b, a).reshape(3,3)
print(c)
```

### 63. Create an array class that has a name attribute (★★☆)

```python
class NamedArray(np.ndarray):
    def __new__(cls, input_array, name=""):
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        return obj

a = NamedArray([1,2,3], name="MyArray")
print(a.name)
```

### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

```python
a = np.zeros(10)
indices = [1,2,2,3]
np.add.at(a, indices, 1)
print(a)
```

### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

Same as 64, use `np.add.at`:

```python
F = np.zeros(10)
X = np.arange(4)
I = [1,2,2,3]
np.add.at(F, I, X)
print(F)
```

### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)

```python
img = np.random.randint(0, 256, (10,10,3), dtype=np.uint8)
colors = img.reshape(-1, 3)
unique_colors = np.unique(colors, axis=0)
print(len(unique_colors))
```

### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

```python
a = np.random.random((2,3,4,5))
s = a.sum(axis=(-1, -2))
print(s.shape)
```

### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)

```python
D = np.random.random(10)
S = np.array([0,1,0,1,0,1,0,1,0,1])
means = np.bincount(S, weights=D) / np.bincount(S)
print(means)
```

### 69. How to get the diagonal of a dot product? (★★★)

```python
A = np.random.random((3,3))
B = np.random.random((3,3))
diag = np.sum(A * B.T, axis=1)
print(diag)
```

### 70. Consider the vector [1][2][3][4][5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

```python
a = np.array([1,2,3,4,5])
b = np.zeros(len(a) + 3*(len(a)-1), dtype=a.dtype)
b[::4] = a
print(b)
```

### 71. Consider an array of dimension (5,5,3), how to multiply it by an array with dimensions (5,5)? (★★★)

```python
a = np.random.random((5,5,3))
b = np.random.random((5,5))
result = a * b[:, :, None]
print(result.shape)
```

### 72. How to swap two rows of an array? (★★★)

```python
a = np.arange(9).reshape(3,3)
a[[0,1]] = a[[1,0]]
print(a)
```

### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★)

```python
triangles = np.random.randint(0, 10, (10, 3))
edges = np.vstack([triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]])
edges = np.sort(edges, axis=1)
unique_edges = np.unique(edges, axis=0)
print(unique_edges)
```

### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

```python
C = np.array([0, 2, 3, 0, 1])
A = np.repeat(np.arange(len(C)), C)
print(np.bincount(A) == C)
```

### 75. How to compute averages using a sliding window over an array? (★★★)

```python
def sliding_window_avg(a, window):
    return np.convolve(a, np.ones(window), 'valid') / window

a = np.arange(10)
print(sliding_window_avg(a, 3))
```

### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z,Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)

```python
Z = np.arange(11)
n = 3
result = np.lib.stride_tricks.sliding_window_view(Z, n)
print(result)
```

### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

```python
a = np.array([True, False, True])
a[:] = ~a
print(a)

b = np.array([1.0, -2.0, 3.0])
b *= -1
print(b)
```

### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)

```python
P0 = np.random.random((5,2))
P1 = np.random.random((5,2))
p = np.array([0.5, 0.5])

def point_line_distance(P0, P1, p):
    line_vec = P1 - P0
    p_vec = p - P0
    line_len = np.linalg.norm(line_vec, axis=1)
    line_unitvec = line_vec / line_len[:, None]
    p_vec_scaled = np.sum(p_vec * line_unitvec, axis=1)
    proj = P0 + line_unitvec * p_vec_scaled[:, None]
    dist = np.linalg.norm(proj - p, axis=1)
    return dist

distances = point_line_distance(P0, P1, p)
print(distances)
```

### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)

```python
P0 = np.random.random((5,2))
P1 = np.random.random((5,2))
P = np.random.random((3,2))

def distances(P0, P1, P):
    line_vec = P1 - P0
    line_len = np.linalg.norm(line_vec, axis=1)
    line_unitvec = line_vec / line_len[:, None]
    dists = np.empty((len(P), len(P0)))
    for i, p in enumerate(P):
        p_vec = p - P0
        p_vec_scaled = np.sum(p_vec * line_unitvec, axis=1)
        proj = P0 + line_unitvec * p_vec_scaled[:, None]
        dists[i] = np.linalg.norm(proj - p, axis=1)
    return dists

print(distances(P0, P1, P))
```

### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary) (★★★)

```python
def extract_subarray(arr, center, shape, fill=0):
    slices = []
    pads = []
    for c, s, max_s in zip(center, shape, arr.shape):
        start = c - s//2
        end = start + s
        pad_before = max(0, -start)
        pad_after = max(0, end - max_s)
        slices.append(slice(max(start,0), min(end, max_s)))
        pads.append((pad_before, pad_after))
    sub = arr[tuple(slices)]
    if any(pad != (0,0) for pad in pads):
        sub = np.pad(sub, pads, constant_values=fill)
    return sub

a = np.arange(25).reshape(5,5)
print(extract_subarray(a, (0,0), (3,3)))
```

### 81. Consider an array Z = [1][2][3][4][5][6][7][8][9][10], how to generate an array R = [[1][2][3][4], [2][3][4][5], [3][4][5][6], ..., ]? (★★★)

```python
Z = np.arange(1,15)
R = np.lib.stride_tricks.sliding_window_view(Z, 4)
print(R)
```

### 82. Compute a matrix rank (★★★)

```python
a = np.random.random((5,5))
rank = np.linalg.matrix_rank(a)
print(rank)
```

### 83. How to find the most frequent value in an array?

```python
a = np.array([1,2,2,3,3,3,4])
values, counts = np.unique(a, return_counts=True)
most_frequent = values[counts.argmax()]
print(most_frequent)
```

### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

```python
import numpy as np

a = np.random.random((10, 10))
shape = (a.shape[0] - 2, a.shape[1] - 2, 3, 3)
strides = a.strides * 2
blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
print(blocks.shape)  # (8, 8, 3, 3)
```

### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)

```python
class SymmetricArray(np.ndarray):
    def __setitem__(self, index, value):
        i, j = index
        super().__setitem__((i, j), value)
        super().__setitem__((j, i), value)

def symmetric_array(input_array):
    obj = np.asarray(input_array).view(SymmetricArray)
    return obj

Z = symmetric_array(np.zeros((3, 3)))
Z[0, 1] = 5
print(Z)
print(Z[1, 0])  # 5
```

### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of the p matrix products at once? (result has shape (n,1)) (★★★)

```python
p, n = 4, 3
matrices = np.random.random((p, n, n))
vectors = np.random.random((p, n, 1))
result = np.sum(np.matmul(matrices, vectors), axis=0)
print(result.shape)  # (n, 1)
```

### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

```python
a = np.arange(16*16).reshape(16,16)
block_size = 4
blocks = a.reshape(16//block_size, block_size, 16//block_size, block_size)
block_sum = blocks.sum(axis=(1,3))
print(block_sum.shape)  # (4,4)
print(block_sum)
```

### 88. How to implement the Game of Life using numpy arrays? (★★★)

```python
def game_of_life_step(X):
    neighbors = sum(np.roll(np.roll(X, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))
    return (neighbors == 3) | (X & (neighbors == 2))

X = np.random.randint(0, 2, (10, 10), dtype=bool)
X = game_of_life_step(X)
print(X)
```

### 89. How to get the n largest values of an array (★★★)

```python
a = np.random.random(10)
n = 3
largest = np.partition(a, -n)[-n:]
largest_sorted = np.sort(largest)[::-1]
print(largest_sorted)
```

### 90. Given an arbitrary number of vectors, build the cartesian product (every combination of every item) (★★★)

```python
from itertools import product

vectors = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
cartesian_product = np.array(list(product(*vectors)))
print(cartesian_product)
```

### 91. How to create a record array from a regular array? (★★★)

```python
a = np.array([(1, 2.), (3, 4.)], dtype=[('x', int), ('y', float)])
rec_arr = a.view(np.recarray)
print(rec_arr.x)
print(rec_arr.y)
```

### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

```python
Z = np.random.random(1000000)
# Method 1: Using **
res1 = Z ** 3
# Method 2: Using np.power
res2 = np.power(Z, 3)
# Method 3: Using multiply
res3 = Z * Z * Z
print(np.allclose(res1, res2) and np.allclose(res2, res3))
```

### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

```python
A = np.random.randint(0, 10, (8, 3))
B = np.array([[1, 2], [3, 4]])

def contains_elements(row, elems):
    return set(elems).issubset(row)

mask = np.array([all(contains_elements(row, b) for b in B) for row in A])
print(A[mask])
```

### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. ) (★★★)

```python
a = np.random.randint(0, 5, (10, 3))
mask = np.any(a != a[:, [0]], axis=1)
unequal_rows = a[mask]
print(unequal_rows)
```

### 95. Convert a vector of ints into a matrix binary representation (★★★)

```python
a = np.array([2, 7, 5])
n_bits = a.max().bit_length()
binary_matrix = ((a[:, None] & (1 << np.arange(n_bits))) > 0).astype(int)
print(binary_matrix)
```

### 96. Given a two dimensional array, how to extract unique rows? (★★★)

```python
a = np.array([[1,2], [1,2], [3,4]])
unique_rows = np.unique(a, axis=0)
print(unique_rows)
```

### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

```python
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

inner = np.einsum('i,i->', A, B)
outer = np.einsum('i,j->ij', A, B)
sum_ = np.einsum('i->', A)
mul = np.einsum('i->', A) * np.einsum('j->', B)

print(inner)
print(outer)
print(sum_)
print(mul)
```

### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

```python
X = np.cumsum(np.random.random(10))
Y = np.cumsum(np.random.random(10))
dist = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
dist = np.insert(dist, 0, 0)
cumdist = np.cumsum(dist)
equidistant = np.linspace(0, cumdist[-1], 50)
X_sampled = np.interp(equidistant, cumdist, X)
Y_sampled = np.interp(equidistant, cumdist, Y)
print(X_sampled, Y_sampled)
```

### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

```python
n = 5
X = np.array([[1,2,2], [2.5, 2, 0.5], [3,1,1]])
mask = (X.sum(axis=1) == n) & np.all(np.mod(X, 1) == 0, axis=1)
valid_rows = X[mask]
print(valid_rows)
```

### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)

```python
X = np.random.random(100)
N = 1000
means = np.array([np.mean(np.random.choice(X, size=len(X), replace=True)) for _ in range(N)])
conf_int = np.percentile(means, [2.5, 97.5])
print(conf_int)
```

