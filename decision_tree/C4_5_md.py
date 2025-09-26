from data import load_watermelonv2alpha_data, load_watermelonv2_data
import numpy as np
from decision_tree.utils import Node, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Implementation of C4.5 algorithm with missing value handling
class C4_5_for_missing_data:
    def __init__(self,):
        super().__init__()

    def fit(self, X, y): # input: pandas dataframe
        self.attrs_dict = {col: X[col].dropna().unique() for col in X.columns}
        w = pd.Series(1, index=y.index) # initial weights
        self.root = self._build_tree(X, y, w)

    def _information_entropy(self, y, w):
        p = w.groupby(y).sum() / w.sum()
        p = p[p > 0] # avoid log(0)
        return -(p * np.log2(p)).sum()
    
    def _information_gain_ratio_metric(self, X, y, w):
        # First filter out the features whose information gain is greater than the average, 
        # then select the one with the highest gain ratio
        info_gain_list = []
        info_gain_ratio_list = []
        for feature in X.columns:
            X_col = X[feature]
            valid_mask = X_col.notnull()
            valid_col = X_col[valid_mask]
            valid_y = y[valid_mask]
            valid_w = w[valid_mask]

            valid_probs = valid_w.sum() / w.sum()

            information_entropy = self._information_entropy(valid_y, valid_w)

            attr_probs = valid_w.groupby(valid_col).sum() / valid_w.sum()    


            cond_entropy = sum(
                p * self._information_entropy(valid_y[valid_col == val], valid_w[valid_col == val])
                for val, p in attr_probs.items()
            )

            intrinsic_value = valid_probs * (-attr_probs * np.log2(attr_probs)).sum()
            feature_info_gain = valid_probs * (information_entropy - cond_entropy)

            feature_info_gain_ratio = feature_info_gain / intrinsic_value if intrinsic_value != 0 else 0

            info_gain_list.append(feature_info_gain)
            info_gain_ratio_list.append(feature_info_gain_ratio)
        
        info_gain_list = np.array(info_gain_list)
        info_gain_ratio_list = np.array(info_gain_ratio_list)
        
        high_gain_feature = np.where(info_gain_list >= np.mean(info_gain_list))[0]
        feature_idx = high_gain_feature[np.argmax(info_gain_ratio_list[high_gain_feature])]
        return X.columns[feature_idx]

    def _build_tree(self, X, y, w): # only for categorical features
        if len(np.unique(y)) == 1: # all samples belong to the same class
            return Node(is_leaf=True, cls_idx=y.iloc[0])
        
        if X.columns.empty or len(X.drop_duplicates()) == 1: # no features left to split on
            majority_class = y.mode()[0]
            return Node(is_leaf=True, cls_idx=majority_class)
        
        best_feature = self._information_gain_ratio_metric(X, y, w) # return column name

        print("Best feature to split on:", best_feature)

        tmp = Node(is_leaf=False, feature=best_feature)
        for attr in self.attrs_dict[best_feature]:
            selected_X, selected_y, selected_w, attr_probs = self._handle_missing_values(X, y, w, best_feature, attr)
            if len(selected_y) == 0:
                majority_class = y.mode()[0]
                tmp.children_weights[attr] = 0.0
                tmp.children[attr] = Node(is_leaf=True, cls_idx=majority_class)
            else:
                tmp.children_weights[attr] = attr_probs
                tmp.children[attr] = self._build_tree(selected_X, selected_y, selected_w)

        
        print("Built node for feature:", best_feature)
            
        return tmp

    def _handle_missing_values(self, X, y, w, feature, attr):
        X_col = X[feature]
        valid_mask = X_col.notnull()
        valid_col = X_col[valid_mask]
        valid_y = y[valid_mask]
        valid_w = w[valid_mask]
        attr_prob = valid_w[valid_col == attr].sum() / valid_w.sum()

        selected_mask = (X_col == attr) | (X_col.isnull())
        attr_valid_X = X[selected_mask].drop(columns=[feature])
        attr_valid_y = y[selected_mask]
        attr_valid_w = w[selected_mask]
        
        attr_valid_w[X_col.isnull()] *= attr_prob
        return attr_valid_X, attr_valid_y, attr_valid_w, attr_prob

    def predict_with_missing_data(self, X):
        y_hat_list = []
        for idx, row in X.iterrows(): # per sample
            y_hat = self._predict_sample(row)
            y_hat_list.append(y_hat)
        
        return y_hat_list

    def _predict_sample(self, sample):
        def dfs(node, weight=1.0):
            if node.is_leaf:
                return {node.cls_idx: weight}

            value = sample[node.feature]

            if pd.notnull(value):
                if value in node.children:
                    return dfs(node.children[value], weight)
                else:
                    return {node.cls_idx: weight}

            # value is missing, traverse all branches
            results = {}
            for attr, child in node.children.items():
                branch_prob = node.children_weights[attr]
                if branch_prob == 0:
                    continue
                sub_result = dfs(child, weight * branch_prob)
                for cls, w in sub_result.items():
                    results[cls] = results.get(cls, 0) + w
            return results

        class_probs = dfs(self.root, 1.0)

        return max(class_probs.items(), key=lambda x: x[1])[0]


if __name__ == "__main__":
    X, y = load_watermelonv2alpha_data()

    model = C4_5_for_missing_data() 
    model.fit(X, y)

    export_graphviz(model.root, title="Titanic Decision Tree (C4.5)")

    y_hat = model.predict_with_missing_data(X)

    acc = accuracy_score(y, y_hat)
    print(f"Accuracy on training set: {acc:.4f}")

    
