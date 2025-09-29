from sklearn.linear_model import LogisticRegression
from data import load_watermelonv3alpha_data
from decision_tree.utils import export_graphviz, MV_Node
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xlim, ylim = None, None
raw_X, raw_y = None, None

class MultiVarDecisionTree:
    def __init__(self,):
        self.root = None
        self.prob_threshold = 0.5

    def fit(self, X, y, draw=False):
        self.node_count = 0  
        self.root = self._build_tree(X, y, draw=draw)

    def draw_classifier(self, X, y, classifier, pclassifier, node_id=0, pnode=None):
        plt.clf()
        plt.scatter(raw_X.loc[raw_y==0, raw_X.columns[0]], raw_X.loc[raw_y==0, raw_X.columns[1]], marker="o", c="gray")
        plt.scatter(raw_X.loc[raw_y==1, raw_X.columns[0]], raw_X.loc[raw_y==1, raw_X.columns[1]], marker="+", c="gray")

        plt.scatter(X.loc[y==0, X.columns[0]], X.loc[y==0, X.columns[1]], marker="o")
        plt.scatter(X.loc[y==1, X.columns[0]], X.loc[y==1, X.columns[1]], marker="+")

        x_min, x_max = X.iloc[:,0].min() - 0.1, X.iloc[:,0].max() + 0.1
        points = np.linspace(x_min, x_max, 100)
        
        w = classifier.coef_[0]
        b = classifier.intercept_[0]
        y_points = -(w[0] * points + b) / w[1]
        
        plt.plot(points, y_points, "b-", label=f"node {node_id}")
        if pclassifier is not None:
            parent_y_points = -(pclassifier.coef_[0][0] * points + pclassifier.intercept_[0]) / pclassifier.coef_[0][1]
            plt.plot(points, parent_y_points, "r-", label=f"pnode {pnode}")

        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
        plt.pause(2)

    def _build_tree(self, X, y, pclassifier=None, pnode=None, draw=False):
        self.node_count += 1
        node_id = self.node_count

        if len(np.unique(y)) == 1:
            return MV_Node(id=node_id, is_leaf=True, cls_idx=y.iloc[0])

        classifier = LogisticRegression(
            penalty=None, solver='lbfgs', max_iter=1000
        )
        classifier.fit(X, y)
        
        if draw:
            self.draw_classifier(X, y, classifier, pclassifier, node_id=node_id, pnode=pnode) 

        proba = classifier.predict_proba(X)[:, 1]
        left_mask = proba <= self.prob_threshold
        right_mask = proba > self.prob_threshold

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return MV_Node(id=node_id, is_leaf=True, cls_idx=y.mode()[0])

        else:
            return MV_Node(
                id=node_id, is_leaf=False, classifier=classifier,
                children={
                    "left": self._build_tree(X[left_mask], y[left_mask], pclassifier=classifier, pnode=node_id, draw=draw),
                    "right": self._build_tree(X[right_mask], y[right_mask], pclassifier=classifier, pnode=node_id, draw=draw)
                })

    def predict(self, X):
        return [self.predict_sample(X.iloc[i:i+1]) for i in range(len(X))]

    def predict_sample(self, sample, node=None):
        if node is None:
            node = self.root
        if node.is_leaf:
            return node.cls_idx
        else:
            proba = node.classifier.predict_proba(sample)[0, 1] # single sample
            if proba <= self.prob_threshold:
                return self.predict_sample(sample, node.children["left"])
            else:
                return self.predict_sample(sample, node.children["right"])


if __name__ == "__main__":
    X, y = load_watermelonv3alpha_data()  # only two continuous features
    y = y.map({"是": 1, "否": 0})
    raw_X, raw_y = X, y
    title = "MultiVarable Decision Tree on Watermelon v3.0 alpha"

    xlim = (X.iloc[:,0].min() - 0.1, X.iloc[:,0].max() + 0.1)
    ylim = (X.iloc[:,1].min() - 0.1, X.iloc[:,1].max() + 0.1)

    plt.ion()
    model = MultiVarDecisionTree()
    model.fit(X, y, draw=True)

    y_hat = model.predict(X)
    acc = accuracy_score(y, y_hat)
    print(f"Accuracy on training set: {acc:.4f}")