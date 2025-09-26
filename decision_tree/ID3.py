from data import load_watermelonv2_data, load_mushroom_data
import numpy as np
from decision_tree.utils import Node, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class ID3:
    def __init__(self, ):
        self.root = None
    
    def fit(self, X, y): # input: pandas dataframe
        self.attrs_dict = {col: X[col].unique() for col in X.columns}
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y): # only for categorical features
        if len(np.unique(y)) == 1: # all samples belong to the same class
            return Node(is_leaf=True, cls_idx=y.iloc[0])
        
        if X.columns.empty or len(X.drop_duplicates()) == 1: # no features left to split on
            majority_class = y.mode()[0]
            return Node(is_leaf=True, cls_idx=majority_class)
        
        best_feature = self.select_best_feature(X, y) # return column name
        print("Best feature to split on:", best_feature)

        tmp = Node(is_leaf=False, feature=best_feature)
        for attr in self.attrs_dict[best_feature]:
            selected_X = X[X[best_feature] == attr].drop(columns=[best_feature])
            selected_y = y[X[best_feature] == attr]
            if len(selected_y) == 0:
                majority_class = y.mode()[0]
                tmp.children[attr] = Node(is_leaf=True, cls_idx=majority_class)
            else:
                tmp.children[attr] = self._build_tree(selected_X, selected_y)
            
        return tmp

    def _information_entropy(self, y):
        p = y.value_counts(normalize=True)
        p = p[p > 0] # avoid log(0)
        return -(p * np.log2(p)).sum()
    
    def select_best_feature(self, X, y):
        return self._information_gain_metric(X, y)
            
    def _information_gain_metric(self, X, y):
        information_entropy = self._information_entropy(y)

        max_info_gain = -np.inf
        best_feature = None

        for feature in X.columns:
            cond_entropy = 0
            for val, idx in X.groupby(feature).groups.items():
                cond_entropy += (len(idx) / len(X)) * self._information_entropy(y.loc[idx])

            feature_info_gain = information_entropy - cond_entropy
            if max_info_gain < feature_info_gain:
                max_info_gain = feature_info_gain
                best_feature = feature
        
        return best_feature
                
    def predict(self, X):
        y_hat_list = []
        for idx, row in X.iterrows(): # per sample
            y_hat = self._predict_sample(row)
            y_hat_list.append(y_hat)
        
        return y_hat_list
    
    def _predict_sample(self, sample):

        node = self.root
        while not node.is_leaf:
            value = sample[node.feature]
            if value not in node.children:
                print(f"Warning: feature value {value} not seen in training for feature {node.feature}. Using majority class.")
                return node.cls_idx
            node = node.children[value]
        return node.cls_idx

        
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        if node.is_leaf:
            print("  " * depth + f"→ 预测: {node.cls_idx}")
        else:
            print("  " * depth + f"[特征: {node.feature}]")
            for attr_val, child in node.children.items():
                print("  " * (depth + 1) + f"如果 {node.feature} = {attr_val}:")
                self.print_tree(child, depth + 2)


if __name__ == "__main__":
    dataset = "watermelon"
    if dataset == "watermelon":
        X, y = load_watermelonv2_data()
        title = "ID3 Decision Tree on Watermelon v2.0"
    elif dataset == "mushroom":
        X, y = load_mushroom_data()
        title = "ID3 Decision Tree on Mushroom Dataset"
    else:
        raise ValueError("Unknown dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = ID3() # information gain
    model.fit(X_train, y_train)
    # export_graphviz(model.root, title=title)

    y_hat = model.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    print(f"Accuracy on testing set: {acc:.4f}")

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_encoded = enc.fit_transform(X_train)
    X_test_encoded = enc.transform(X_test)

    sklearn_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    sklearn_model.fit(X_train_encoded, y_train)
    y_sklearn_hat = sklearn_model.predict(X_test_encoded)
    sklearn_acc = accuracy_score(y_test, y_sklearn_hat)
    print(f"Sklearn DecisionTreeClassifier Accuracy on testing set: {sklearn_acc:.4f}")