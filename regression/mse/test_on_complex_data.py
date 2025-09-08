from sklearn.datasets import make_regression
from sklearn.datasets import fetch_california_housing
from regression.mse.mse_regulation import MSE_Solver
import numpy as np

# test 1 on synthetic data
X, y, coef = make_regression(
    n_samples=100, 
    n_features=20, 
    n_informative=5,  # 仅 5 个特征有用
    noise=10.0,        # 加入噪声
    coef=True, 
    random_state=42
)

mse = MSE_Solver()

X = (X-X.mean(axis=0)) / X.std(axis=0)  # 标准化
X = np.hstack((X, np.ones((X.shape[0], 1))))
y = y.reshape(-1, 1)
print(coef)

mse.Lasso_fit(X, y, 10)
print(mse.W.flatten())
lasso_y_hat = mse.predict(X)

loss = np.mean((y - lasso_y_hat)**2) / X.shape[0]
print("Lasso MSE loss:", loss)

mse.ridge_fit(X, y)
print(mse.W.flatten())

ridge_y_hat = mse.predict(X)

loss = np.mean((y - ridge_y_hat)**2) / X.shape[0]
print("ridge MSE loss:", loss)

# test 2 on california data
housing = fetch_california_housing()
X, y = housing.data, housing.target
y = y.reshape(-1, 1)
X = np.hstack((X, np.ones((X.shape[0], 1))))

mse.Lasso_fit(X, y, 10)
print(mse.W.flatten())
lasso_y_hat = mse.predict(X)

loss = np.mean((y - lasso_y_hat)**2) / X.shape[0]
print("(House) Lasso MSE loss:", loss)