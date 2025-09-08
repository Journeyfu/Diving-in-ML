from sklearn.datasets import make_regression

X, y, coef = make_regression(
    n_samples=100, 
    n_features=20, 
    n_informative=5,  # 仅 5 个特征有用
    noise=10.0,        # 加入噪声
    coef=True, 
    random_state=42
)
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
y = boston.target