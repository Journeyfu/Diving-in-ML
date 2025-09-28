from data import load_watermelonv3_data
import numpy as np
from decision_tree.utils import Node, export_graphviz
from sklearn.metrics import accuracy_score

# Implementation of C4.5 algorithm with continuous feature handling, following the Machine Learning by Zhou Zhihua
class C4_5_for_continguous_data():
    def __init__(self, ):
        super().__init__()

    def fit(self, X, y, continguous_features): # input: pandas dataframe
        self.categorical_attrs_dict = {col: X[col].unique() for col in X.columns if col not in continguous_features}
        self.continguous_features = continguous_features
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y): # only for categorical features
        if len(np.unique(y)) == 1: # all samples belong to the same class
            return Node(is_leaf=True, cls_idx=y.iloc[0])
        
        if X.columns.empty or len(X.drop_duplicates()) == 1: # no features left to split on
            majority_class = y.mode()[0]
            return Node(is_leaf=True, cls_idx=majority_class)
        
        best_feature, threshold = self.select_best_feature(X, y) # return column name
        print("Best feature to split on:", best_feature)

        tmp = Node(is_leaf=False, feature=best_feature)
        if best_feature in self.continguous_features:
            tmp.threshold = threshold

            selected_X = X[X[best_feature] <= threshold] # keep the feature for the children
            selected_y = y[X[best_feature] <= threshold]
            attr = f"{best_feature}<={threshold:.3f}"
            if len(selected_y) == 0:
                majority_class = y.mode()[0]
                tmp.children[attr] = Node(is_leaf=True, cls_idx=majority_class)
            else:
                tmp.children[attr] = self._build_tree(selected_X, selected_y)
            
            selected_X = X[X[best_feature] > threshold]
            selected_y = y[X[best_feature] > threshold]
            attr = f"{best_feature}>{threshold:.3f}"
            if len(selected_y) == 0:
                majority_class = y.mode()[0]
                tmp.children[attr] = Node(is_leaf=True, cls_idx=majority_class)
            else:
                tmp.children[attr] = self._build_tree(selected_X, selected_y)
            
        else:
            for attr in self.categorical_attrs_dict[best_feature]:
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
        return self._information_gain_ratio_metric(X, y)
    
    def _get_information_gain_for_continuous(self, X_col, y, information_entropy):
        col = X_col.sort_values().to_numpy()
        continguous_threshold = (col[:-1] + col[1:]) / 2.
        continguous_threshold = np.unique(continguous_threshold)

        max_info_gain_ratio = -np.inf
        max_info_gain = 0.
        return_threshold = None
        for threshold in continguous_threshold:
            binary_col = X_col <= threshold
            feature_info_gain, feature_info_gain_ratio = self._get_information_gain_for_categorical(binary_col, y, information_entropy)
            if feature_info_gain_ratio > max_info_gain_ratio:
                max_info_gain_ratio = feature_info_gain_ratio
                max_info_gain = feature_info_gain
                return_threshold = threshold 
        
        return max_info_gain, max_info_gain_ratio, return_threshold

    def _get_information_gain_for_categorical(self, X_col, y, information_entropy):
        probs = X_col.value_counts(normalize=True)
        cond_entropy = sum(
            p * self._information_entropy(y[X_col == val])
            for val, p in probs.items()
        )
        intrinsic_value = -(probs * np.log2(probs)).sum()
        feature_info_gain = information_entropy - cond_entropy

        feature_info_gain_ratio = feature_info_gain / intrinsic_value if intrinsic_value != 0 else 0
        return feature_info_gain, feature_info_gain_ratio

    def _information_gain_ratio_metric(self, X, y):
        # First filter out the features whose information gain is greater than the average, 
        # then select the one with the highest gain ratio
        information_entropy = self._information_entropy(y)

        info_gain_list = []
        info_gain_ratio_list = []
        threshold_list = []
        for feature in X.columns:

            if feature in self.continguous_features: # preprocess continuous features
                feature_info_gain, feature_info_gain_ratio, threshold = self._get_information_gain_for_continuous(X[feature], y, information_entropy)
            else:
                feature_info_gain, feature_info_gain_ratio = self._get_information_gain_for_categorical(X[feature], y, information_entropy)

            info_gain_list.append(feature_info_gain)
            info_gain_ratio_list.append(feature_info_gain_ratio)
            threshold_list.append(threshold if feature in self.continguous_features else None)
        
        info_gain_list = np.array(info_gain_list)
        info_gain_ratio_list = np.array(info_gain_ratio_list)
        
        high_gain_feature = np.where(info_gain_list >= np.mean(info_gain_list))[0]
        feature_idx = high_gain_feature[np.argmax(info_gain_ratio_list[high_gain_feature])]
        return X.columns[feature_idx], threshold_list[feature_idx]

    def predict(self, X):
        y_hat_list = []
        for idx, row in X.iterrows(): # per sample
            y_hat = self._predict_sample(row)
            y_hat_list.append(y_hat)
        
        return y_hat_list
    
    def _predict_sample(self, sample):

        node = self.root
        while not node.is_leaf:
            if node.feature in self.continguous_features:
                threshold = node.threshold
                value = f"{node.feature}<={threshold:.3f}" if sample[node.feature] <= threshold else f"{node.feature}>{threshold:.3f}"
            else:
                value = sample[node.feature]

            if value not in node.children:
                print(f"Warning: feature value {value} not seen in training for feature {node.feature}. Using majority class.")
                return node.cls_idx
            node = node.children[value]
        return node.cls_idx

        
if __name__ == "__main__":
    X, y = load_watermelonv3_data()
    title = "C4.5 Decision Tree on Watermelon v3.0"

    model = C4_5_for_continguous_data() # information gain ratio
    model.fit(X, y, continguous_features=X.columns.tolist()[-2:])
    export_graphviz(model.root, title=title)

    y_hat = model.predict(X)

    acc = accuracy_score(y, y_hat)
    print(f"Accuracy on training set: {acc:.4f}")