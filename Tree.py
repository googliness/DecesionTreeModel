class Node:
    def __init__(self, score=0, isLeaf=False, left=None, right=None, featureIndex=-1):
        self.isLeaf = isLeaf
        self.left = left
        self.right = right
        self.score = score
        self.featureIndex = featureIndex

