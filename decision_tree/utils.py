import numpy as np
from graphviz import Digraph
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
from io import BytesIO

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False

class Node:
    def __init__(self, feature=None, is_leaf=False, cls_idx=None):
        self.feature = feature # feature name to split on
        self.is_leaf = is_leaf 
        self.cls_idx = cls_idx # is only valid when is_leaf is True
        self.children = {}


def export_graphviz(root, title="ID3 Decision Tree"):
    import os
    os.environ["PATH"] += os.pathsep + r"C:\apps\Graphviz\bin"
    dot = Digraph(comment="ID3 Decision Tree", format="png")

    def add_nodes_edges(node, parent=None, edge_label=""):
        if node.is_leaf:
            node_id = str(id(node))
            label = f"预测: {node.cls_idx}"
            dot.node(node_id, label, shape="box", style="filled", color="lightgrey", fontname="Microsoft YaHei")
        else:
            node_id = str(id(node))
            dot.node(node_id, f"特征: {node.feature}", fontname="Microsoft YaHei")

        if parent:
            dot.edge(parent, node_id, label=edge_label, fontname="Microsoft YaHei")

        if not node.is_leaf:
            for attr_val, child in node.children.items():
                add_nodes_edges(child, node_id, str(attr_val))

    add_nodes_edges(root)
    img_bytes = dot.pipe(format="png")

    img = mpimg.imread(BytesIO(img_bytes), format="png")
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()