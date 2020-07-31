import numpy as np  # Imports
import torch

X = np.random.rand(30, 1)*2.0  # Create data as numpy arrays
w = np.random.rand(2, 1)  # Randomly select weights of linear regressor
y = X*w[0] + w[1] + np.random.randn(30, 1) * 0.05    # Create targets
print('target w {} b {}'.format(w[0], w[1]))
Xt = torch.from_numpy(X).float()  # Convert numpy arrays to torch tensors
yt = torch.from_numpy(y).float()

W = torch.rand(1, requires_grad=True)  # Initialize weights randomly with parameter requires_grad=True
b = torch.rand(1, requires_grad=True)

lr = 0.005  # set up learning rate
for epoch in range(2500):
    y_pred = torch.add(torch.mul(W,Xt), b)  # W*x + b    # Compute predictions
    loss = torch.mean((y_pred - yt) ** 2)   # Compute cost function
    loss.backward()   # Run back-propagation
    W.data = W.data - lr*W.grad.data   # Update variables
    b.data = b.data - lr*b.grad.data
    W.grad.data.zero_()   # Reset gradients
    b.grad.data.zero_()

print('found w {} b {}'.format(W.data ,b.data))