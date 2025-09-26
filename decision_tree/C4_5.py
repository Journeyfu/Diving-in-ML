
from data import load_watermelonv2_data, load_mushroom_data
import numpy as np
from decision_tree.utils import Node, export_graphviz
from decision_tree.ID3 import ID3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class C4_5(ID3):
    def __init__(self, ):
        super().__init__()

    def select_best_feature(self, X, y):
        return self._information_gain_ratio_metric(X, y)

    def _information_gain_ratio_metric(self, X, y):
        # First filter out the features whose information gain is greater than the average, 
        # then select the one with the highest gain ratio
        information_entropy = self._information_entropy(y)

        info_gain_list = []
        info_gain_ratio_list = []
        for feature in X.columns:
            probs = X[feature].value_counts(normalize=True)
            cond_entropy = sum(
                p * self._information_entropy(y[X[feature] == val])
                for val, p in probs.items()
            )
            intrinsic_value = -(probs * np.log2(probs)).sum()
            feature_info_gain = information_entropy - cond_entropy

            feature_info_gain_ratio = feature_info_gain / intrinsic_value if intrinsic_value != 0 else 0

            info_gain_list.append(feature_info_gain)
            info_gain_ratio_list.append(feature_info_gain_ratio)
        
        info_gain_list = np.array(info_gain_list)
        info_gain_ratio_list = np.array(info_gain_ratio_list)
        
        high_gain_feature = np.where(info_gain_list >= np.mean(info_gain_list))[0]
        feature_idx = high_gain_feature[np.argmax(info_gain_ratio_list[high_gain_feature])]
        return X.columns[feature_idx]
        
if __name__ == "__main__":
    dataset = "mushroom"  # "watermelon" or "mushroom"
    if dataset == "watermelon":
        X, y = load_watermelonv2_data()
        title = "C4.5 Decision Tree on Watermelon v2.0"
    elif dataset == "mushroom":
        X, y = load_mushroom_data()
        title = "C4.5 Decision Tree on Mushroom Dataset"
    else:
        raise ValueError("Unknown dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = C4_5() # information gain ratio
    model.fit(X_train, y_train)
    # model.print_tree()
    export_graphviz(model.root, title=title)

    y_hat = model.predict(X_test)

    acc = accuracy_score(y_test, y_hat)
    print(f"Accuracy on testing set: {acc:.4f}")