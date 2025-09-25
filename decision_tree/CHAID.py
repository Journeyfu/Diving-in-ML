from data import load_watermelonv2_data, load_mushroom_data
import numpy as np
from decision_tree.utils import Node, export_graphviz
from decision_tree.ID3 import ID3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.stats import chi2_contingency
from itertools import combinations

# Chi-squared Automatic Interaction Detector
class CHAID(ID3):
    def __init__(self, ):
        super().__init__()

    def select_best_feature(self, X, y):
        return self.chi_square_score(X, y)

    def chi_square_score(self, X, y, alpha=0.05):
        min_p_value = 1.0 # p-value ranges from 0 to 1
        selected_feature = None
        X_new = None
        for feature in X.columns:
            X_col = X[feature].copy()
            while True:
                attrs = X_col.unique()
                pairs = list(combinations(attrs, 2)) 
                max_p_value = -1.0
                merge_attrs = None
                num_pairs = len(pairs)
                corrected_alpha = alpha / num_pairs if num_pairs > 0 else alpha # Bonferroni correction
                for f1, f2 in pairs: # merge two feature attributes with least significant difference
                    mask = X_col.isin([f1, f2])
                    contingency_table = pd.crosstab(X_col[mask], y[mask])
                    chi2, p, dof, ex = chi2_contingency(contingency_table)
                    if p > max_p_value:
                        max_p_value = p
                        merge_attrs = (f1, f2)
                if max_p_value > corrected_alpha:
                    X_col = X_col.replace({merge_attrs[0]: f"{merge_attrs[0]}_{merge_attrs[1]}",
                                                     merge_attrs[1]: f"{merge_attrs[0]}_{merge_attrs[1]}"})
                else:
                    break
            
            contingency_table = pd.crosstab(X_col, y)
            chi2, p, dof, ex = chi2_contingency(contingency_table)

            # Only consider when significant
            if p < min_p_value and p < alpha:
                min_p_value = p
                selected_feature = feature
                X_new = X.copy()
                X_new[feature] = X_col
        
        if X_new is not None:
            self.attrs_dict[selected_feature] = X_new[selected_feature].unique()
            return selected_feature, X_new
        else:
            return None, None
            

    def _build_tree(self, X, y): # only for categorical features
        if len(np.unique(y)) == 1: # all samples belong to the same class
            return Node(is_leaf=True, cls_idx=y.iloc[0])
        
        if X.columns.empty or len(X.drop_duplicates()) == 1: # no features left to split on
            majority_class = y.mode()[0]
            return Node(is_leaf=True, cls_idx=majority_class)
        
        best_feature, X = self.select_best_feature(X, y) # return column name
        
        if best_feature is None: # No feature is significant
            return Node(is_leaf=True, cls_idx=y.mode()[0])

        majority_class = y.mode()[0]
        tmp = Node(is_leaf=False, cls_idx=majority_class,  feature=best_feature)
        for attr in self.attrs_dict[best_feature]:
            selected_X = X[X[best_feature] == attr].drop(columns=[best_feature])
            selected_y = y[X[best_feature] == attr]
            if len(selected_y) == 0:
                majority_class = y.mode()[0]
                tmp.children[attr] = Node(is_leaf=True, cls_idx=majority_class)
            else:
                tmp.children[attr] = self._build_tree(selected_X, selected_y)
            
        return tmp

    def _predict_sample(self, sample):

        node = self.root
        while not node.is_leaf:
            value = sample[node.feature]
            for key in node.children.keys():
                if value in key.split('_'):
                    value = key
                    break
            if value not in node.children:
                print(f"Warning: feature value {value} not seen in training for feature {node.feature}. Using majority class.")
                return node.cls_idx
            node = node.children[value]
        return node.cls_idx

        
if __name__ == "__main__":
    dataset = "mushroom"
    if dataset == "watermelon":
        X, y = load_watermelonv2_data()
        title = "CHAID Decision Tree on Watermelon v2.0"
    elif dataset == "mushroom":
        X, y = load_mushroom_data()
        title = "CHAID Decision Tree on Mushroom Dataset"
    else:
        raise ValueError("Unknown dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CHAID() # Gini index
    model.fit(X_train, y_train)
    # model.print_tree()
    export_graphviz(model.root, title=title)

    y_hat = model.predict(X_test)

    acc = accuracy_score(y_test, y_hat)
    print(f"Accuracy on testing set: {acc:.4f}")



