import numpy as np
from graphviz import Digraph
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import os
from io import BytesIO


matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False

class Node: # single variable node
    def __init__(self, feature=None, is_leaf=False, cls_idx=None):
        self.feature = feature # feature name to split on
        self.is_leaf = is_leaf 
        self.cls_idx = cls_idx # is only valid when is_leaf is True
        self.children = {}
        self.children_weights = {}
        self.threshold = None

class MV_Node: # multivariate node
    def __init__(self, id=None, is_leaf=False, cls_idx=None, classifier=None, children={}):
        self.id = id
        self.cls_idx = cls_idx
        self.is_leaf = is_leaf 
        self.classifier = classifier # linear classifier
        self.children = children

def export_graphviz(root, title="C4.5 Decision Tree", bin_path=r"C:\apps\Graphviz\bin"):

    os.environ["PATH"] += os.pathsep + bin_path
    dot = Digraph(comment=title, format="png")

    def add_nodes_edges(node, parent=None, edge_label=""):
        node_id = str(id(node))

        if node.is_leaf:
            label = f"预测: {node.cls_idx}"
            dot.node(node_id, label, shape="box", style="filled", color="lightgrey", fontname="Microsoft YaHei")
        else:
            dot.node(node_id, f"特征: {node.feature}", fontname="Microsoft YaHei")

        if parent:
            dot.edge(parent, node_id, label=edge_label, fontname="Microsoft YaHei")

        if not node.is_leaf:
            for attr_val, child in node.children.items():
                if hasattr(node, "children_weights") and node.children_weights:
                    weight = node.children_weights.get(attr_val, None)
                    if weight is not None:
                        edge_label = f"{attr_val} (w={weight:.2f})"
                    else:
                        edge_label = str(attr_val)
                else:
                    edge_label = str(attr_val)

                add_nodes_edges(child, node_id, edge_label)

    add_nodes_edges(root)

    img_bytes = dot.pipe(format="png")
    img = mpimg.imread(BytesIO(img_bytes), format="png")
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()
