import numpy as np

''' 
backpropagation uses chain rule under the hood to calculate the gradient of the output of a complex function with respect to its inputs.
generally, backpropagation is simplest to visualize as a chained product of gradients of intermediate variables, represented by a computational graph

note "*" is the element-wise multiplier operator and "@" denotes standard matrix multiplication
'''

W = np.random.randn(3, 3)                   # 3x3
x = np.random.randn(3, 1)                   # 3x1

## forward pass to find f (output)
m = W @ x                                   # 3x3 x 3x1   => 3x1
p = np.maximum(0, m)                        # max(0, 3x1) => 3x1
f = np.linalg.norm(p) ** 2                  # sum of squared values => 1x1 (scalar)

## gradients for backward pass
df_dp = 2 * p                               # 2 * 3x1 => 3x1 
dp_dm = (m > 0).astype(float)               # element-wise ReLU: 3x1

dm_dW = x.T                                 # 1x3
df_dW =  (df_dp * dp_dm) @ dm_dW            # (3x1 * 3x1) x 1x3 = 3x3
                                            # note: the ReLU function is applied element-wise so we use the * operator here.
                                            # however, each element of m is affected the sum of different proportional components of W
                                            # the matrix multiplier "@" operator captures this behavior

## backward pass
dm_dx = W
df_dx =  dm_dx @ (dp_dm * df_dp)

print(f"\n-------------------- W, x -------------------- \nW=\n {W},\n\nx=\n {x}", "\n\n-------------------- W, x --------------------\n")
print(f"\n-------------------- df/dW -------------------- \n\n {df_dW}", "\n\n-------------------- df/dW --------------------\n")
print(f"\n-------------------- df/dx -------------------- \n\n {df_dx}", "\n\n-------------------- df/dx --------------------\n\n")
