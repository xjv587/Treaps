from __future__ import annotations
import random
import typing
import math
from collections.abc import Iterator
from typing import List, Optional, cast

from py_treaps.treap import KT, VT, Treap
from py_treaps.treap_node import TreapNode

class TreapMap(Treap[KT, VT]):
    def __init__(self):
        self.root: Optional[TreapNode] = None

    def get_root_node(self) -> Optional[TreapNode]:
        return self.root

    def lookup(self, key: KT) -> Optional[VT]:
        current_node = self.root
        while current_node is not None:
            if key == current_node.key:
                return current_node.value
            elif key < current_node.key:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        return None

    def _left_rotate(self, x: TreapNode) -> None:
        y = x.right_child
        x.right_child = y.left_child
        if y.left_child is not None:
            y.left_child.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left_child:
            x.parent.left_child = y
        else:
            x.parent.right_child = y
        y.left_child = x
        x.parent = y

    def _right_rotate(self, y: TreapNode) -> None:
        x = y.left_child
        y.left_child = x.right_child
        if x.right_child is not None:
            x.right_child.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.left_child:
            y.parent.left_child = x
        else:
            y.parent.right_child = x
        x.right_child = y
        y.parent = x

    def insert(self, key: KT, value: VT) -> None:
        # Search for the node with the given key
        node = self.root
        parent = None
        while node is not None and node.key != key:
            parent = node
            node = node.left_child if key < node.key else node.right_child

        if node is not None:
            node.value = value

        # If no node with the key is found, insert a new node
        else:
            new_node = TreapNode(key, value, parent)
            if parent is None:
                self.root = new_node
            elif key < parent.key:
                parent.left_child = new_node
            else:
                parent.right_child = new_node

            # Maintain heap property
            node = new_node
            while parent is not None and node.priority > parent.priority:
                if parent.left_child == node:
                    self._right_rotate(parent)
                else:
                    self._left_rotate(parent)
                parent = node.parent

    def remove(self, key: KT) -> Optional[VT]:
        # Find the node
        node = self.root
        while node is not None and node.key != key:
            node = node.left_child if key < node.key else node.right_child

        if node is None:  # Key not found
            return None

        # Rotate node down until it is a leaf
        while node.left_child is not None or node.right_child is not None:
            if node.left_child is None:
                self._left_rotate(node)
            elif node.right_child is None:
                self._right_rotate(node)
            elif node.left_child.priority < node.right_child.priority:
                self._right_rotate(node)
            else:
                self._left_rotate(node)

        # Remove the leaf node
        if node.parent is not None:
            if node == node.parent.left_child:
                node.parent.left_child = None
            else:
                node.parent.right_child = None
        else:
            self.root = None

        return node.value

    def split(self, threshold: [KT]) -> List[Treap[KT, VT]]:
        # Find the median key index
        keys = list(self)
        median_index = len(keys) // 2 - 1

        left_treap = TreapMap()
        right_treap = TreapMap()

        # Insert a node with the median key and threshold if there is no median-key node
        if median_index not in keys:
            self.insert(threshold, median_index)

        # Assign the left and right children of the root to the new treaps
        left_treap.root = self.root.left_child
        if left_treap.root is not None:
            left_treap.root.parent = None

        right_treap.root = self.root.right_child
        if right_treap.root is not None:
            right_treap.root.parent = None

        return [left_treap, right_treap]

    def join(self, other: Treap[KT, VT]) -> None:
        # Create a new root node with an arbitrary key and max priority
        new_root = TreapNode(key=None, value=None)
        new_root.priority = -1  # Maximum priority

        # Assign the two treaps to be the left and right children of the new root
        new_root.left_child = self.root
        new_root.right_child = other.root
        if self.root is not None:
            self.root.parent = new_root
        if other.root is not None:
            other.root.parent = new_root

        self.root = new_root

        self.remove(None)

    def meld(self, other: Treap[KT, VT]) -> None:
        # Insert all nodes of the other treap into the current treap
        for key in other:
            self.insert(key, other.lookup(key))

    def difference(self, other: Treap[KT, VT]) -> None:
        # Iterating through T2 requires O(m) time
        # Removing keys from T1, since the size keeps decreasing, so it's O(log(n/m))
        # That's why the overall is O(m log(n/m)) time
        for key in other:
            self.remove(key)

    def balance_factor(self) -> float:
        # Calculate balance factor: actual height / log2(number of nodes + 1)
        def tree_height(node):
            if node is None:
                return 0
            return 1 + max(tree_height(node.left_child), tree_height(node.right_child))

        def count_nodes(node):
            if node is None:
                return 0
            return 1 + count_nodes(node.left_child) + count_nodes(node.right_child)

        n = count_nodes(self.root)
        actual_height = tree_height(self.root)
        min_height = math.log2(n + 1)

        return actual_height / min_height

    def __str__(self) -> str:
        # Generate a pre-order traversal string representation of the treap
        def pre_order_traversal(node):
            if node is None:
                return ""
            left_str = pre_order_traversal(node.left_child)
            right_str = pre_order_traversal(node.right_child)
            return f"[{node.priority}] <{node.key}, {node.value}>\n{left_str}{right_str}"

        return pre_order_traversal(self.root).strip()

    def __iter__(self) -> Iterator[KT]:
        # Generate an in-order traversal iterator of the treap keys
        def in_order_traversal(node):
            if node is not None:
                yield from in_order_traversal(node.left_child)
                yield node.key
                yield from in_order_traversal(node.right_child)

        return iter(in_order_traversal(self.root))

