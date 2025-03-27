from Neural_Network import *

input_size = 5
hidden_size1 = 4
hidden_size2 = 4
output_size = 1
lr = 0.1

NN = Neural_Network(input_size,hidden_size1,hidden_size2,output_size)

import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Generate circular data
X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)

# If you need exactly 5 features, you can add 3 more random features to X
X = np.hstack([X, np.random.randn(X.shape[0], 3)])

# Show the shape of the dataset
print("Features shape:", X.shape)
print("Output shape:", y.shape)

# Plot the data (first two features for visualization purposes)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.title('Generated Circular Non-Linear Data (make_circles)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()



epochs = 10000
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        inputs = X[i].reshape(-1,1)
        target = y[i]
        outputs = NN.Forward(inputs)

        mse = 0.5 * np.mean((outputs - target)**2)

        total_loss += mse

        grad_output = (outputs - target)

        NN.Backward(grad_output,lr)
    if epoch % 100 == 0:
        print(f'{epoch}th epoch loss is {total_loss}')
    
cnt = 0
for i in range(len(X)):
    output = NN.Forward(X[i].reshape(-1,1))
    if output >= 0.9:
        output = 1
    else:
        output = 0
    if output == y[i]:
        cnt += 1
    print(f'predicted val = {output} target is {y[i]}')

print(f'model accuracy is: {(cnt//len(X))*100}')


