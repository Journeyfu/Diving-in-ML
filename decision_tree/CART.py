from data import load_watermelonv2_data, load_mushroom_data
import numpy as np
from decision_tree.utils import Node, export_graphviz
from decision_tree.ID3 import ID3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class CART(ID3):
    def __init__(self, ):
        super().__init__()

    def select_best_feature(self, X, y):
        return self._gini_index_metric(X, y)

    def _gini_index_metric(self, X, y):
        min_gini_index = float("inf")
        selected_feature = None
        for feature in X.columns:
            probs = X[feature].value_counts(normalize=True)
            gini_index = sum(
                p * (1 - (y[X[feature] == val].value_counts(normalize=True)**2).sum())
                for val, p in probs.items()
            )
            if gini_index < min_gini_index:
                min_gini_index = gini_index
                selected_feature = feature
        return selected_feature

        
if __name__ == "__main__":
    dataset = "mushroom"
    if dataset == "watermelon":
        X, y = load_watermelonv2_data()
        title = "CART Decision Tree on Watermelon v2.0"
    elif dataset == "mushroom":
        X, y = load_mushroom_data()
        title = "CART Decision Tree on Mushroom Dataset"
    else:
        raise ValueError("Unknown dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CART() # Gini index
    model.fit(X_train, y_train)
    # model.print_tree()
    export_graphviz(model.root, title=title)

    y_hat = model.predict(X_test)

    acc = accuracy_score(y_test, y_hat)
    print(f"Accuracy on testing set: {acc:.4f}")



