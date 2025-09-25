import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from ucimlrepo import fetch_ucirepo


def get_2d_mixture_data(n, n_ctr, seed=42):
    """
    imput:
        n: number of samples per center
        n_ctr: number of centers
        seed: random seed
    return:  
        n x 2 array
    """
    rng = np.random.default_rng(seed)
    X, labels = [], []
    for i in range(n_ctr):
        mu = rng.uniform(-5, 5, 2)
        A = rng.uniform(-1, 1, (2, 2))
        cov = A @ A.T
        X.append(rng.multivariate_normal(mu, cov, n))
        labels += [i] * (n)
    
    X = np.vstack(X)
    labels = np.array(labels)
    
    return X, labels

def get_x_square_curve_data(n, seed=42):
    """
    imput:
        n: number of samples
        seed: random seed
    return:  
        n x 2 array
    """
    rng = np.random.default_rng(seed)
    # X = rng.uniform(-1, 1, (n,))
    X = np.linspace(-1, 1, n)
    Y = X**2 + rng.normal(0, 0.05, (n,))
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    return X, Y

def get_log_curve_data(n, seed=42):
    """
    imput:
        n: number of samples
        seed: random seed
    return:  
        n x 2 array
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(0.2, 4, n)
    Y = np.log(X) + rng.normal(0, 0.05, (n,))
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    return X, Y

def get_2d_noise_linear_data(n, outlier_fraction=0, seed=42):
    """
    imput:
        n: number of samples
        seed: random seed
    return:  
        n x 2 array
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-100, 100, (n,))
    Y = 0.3 * X + 5 +  rng.normal(0, 5, (n,))
    n_outliers = int(n * outlier_fraction)
    outlier_indices = rng.choice(n, n_outliers, replace=False)
    
    # 给 outliers 加上大幅偏移
    Y[outlier_indices] = rng.uniform(-100, -50, n_outliers)  
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    return X, Y

def get_3d_noise_linear_data(n, outlier_fraction=0, seed=42):
    """
    Returns:
        X: n x 2 array
        y: n x 1 array
    """
    rng = np.random.default_rng(seed)
    
    # 两个特征
    X1 = rng.uniform(-50, 50, n)
    X2 = rng.uniform(-50, 50, n)
    
    # X1 = np.linspace(-50, 50, n)
    # X2 = np.linspace(-50, 50, n)
    
    # 线性关系：y = 2*X1 - 3*X2 + 10 + 高斯噪声
    y = 2*X1 - 3*X2 + 10 + rng.normal(0, 10, n)
    
    # 添加异常点
    n_outliers = int(n * outlier_fraction)
    if n_outliers > 0:
        outlier_indices = rng.choice(n, n_outliers, replace=False)
        y[outlier_indices] = rng.uniform(-100, 10, n_outliers)
    
    # 返回特征矩阵和目标
    X = np.vstack([X1, X2]).T
    y = y.reshape(-1, 1)
    return X, y


def get_onehot(y):
    num_classes = np.max(y) + 1
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot

def load_iris_data(): # all continuous features
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def load_mushroom_data(): 
    """
    all categorical features, including 8124 samples
    2 classes: edible(e) and posionous(p)
    return: pandas DataFrame
    """
    mushroom = fetch_ucirepo(id=73)
    X = mushroom.data.features
    y = mushroom.data.targets["poisonous"]
    return X, y 


def load_watermelonv2_data():
    # Create Zhou Zhihua Watermelon Dataset 2.0 (categorical features) as a pandas DataFrame
    data = [
        ("青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"),
        ("乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"),
        ("乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"),
        ("青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"),
        ("浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"),
        ("青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", "是"),
        ("乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", "是"),
        ("乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", "是"),

        ("乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", "否"),
        ("青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", "否"),
        ("浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", "否"),
        ("浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", "否"),
        ("青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", "否"),
        ("浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", "否"),
        ("乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", "否"),
        ("浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", "否"),
        ("青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", "否"),
    ]
    data = pd.DataFrame(data, columns=["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "好瓜"])
    X = data.drop(columns=["好瓜"])   # 特征
    y = data["好瓜"]                 # 标签
    return X, y

if __name__ == "__main__":
    # Example usage
    """
    X, labels = get_2d_mixture_data(300, 3, seed = 42)
    for i in np.unique(labels):
        plt.scatter(X[labels==i, 0], X[labels==i, 1], s=10, label=f'Cluster {i}')
    plt.title("mm data")
    plt.legend()
    plt.axis("equal")
    plt.show()

    X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    for i in np.unique(labels):
        plt.scatter(X[labels==i, 0], X[labels==i, 1], s=10, label=f'Normed Cluster {i}')
    plt.title("normed mm data")
    plt.legend()
    plt.axis("equal")
    plt.show()
    
    X = get_x_square_curve_data(1000, seed=42)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.axis("equal")
    plt.show()

    X, Y = get_2d_noise_linear_data(1000, outlier_fraction=0.01, seed=42)
    plt.scatter(X, Y, s=10)
    plt.show()

    X, Y = get_3d_noise_linear_data(1000, outlier_fraction=0.01, seed=42)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], Y, s=50)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    ax.set_title("3D visualization of data")
    plt.show()
    """

    X, Y = get_log_curve_data(100, seed=42)
    plt.scatter(X, Y, s=10)
    plt.show()
