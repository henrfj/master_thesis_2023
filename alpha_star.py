"""A* algorithm and some problems to be solved by it.
I used the pseudo-code of 'Essentials of the A* algorithm' published on blackboard"""
__author__ = "Henrik Fjellheim"

import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('classic')


class Node:
    """A search node, in this case representing a position on a board. They have parents and children"""
    def __init__(self, best_parent=None, position=None):
        """
        Initialize the node. Nodes are born without parents. The optimal parent is added later: like adoption
        :param position: is the x and y coordinate of the node
        """
        self.position = position
        self.children = []
        self.best_parent = best_parent
        self.g = 0
        self.h = 0

    def get_f(self):
        """Estimate total cost of solution path through this node"""
        return self.g + self.h

    def __eq__(self, other):
        """We overload eq so that we can compare two nodes"""
        return self.position == other.position

    def __gt__(self, other):
        """We overload gt so that we can sort the list based on f - value"""
        return self.get_f() > other.get_f()

    def __str__(self):
        return "|Position: " + str(self.position) + " f-value: " + str(self.get_f()) + "| "


def a_star(board, start, end):
    """
    Algorithm based on the 'essentials of A* pseudo-code'
    :param board: an integer double list of the maze. first index is x - , second is y - coordinates
    :param start: starting position as a list of two elements [x, y]
    :param end: goal position as a list of two elements [x, y]
    :return: A list og tuple-coordinates, describing the optimal path found
    """
    # Initiate
    closed_nodes = []
    open_nodes = []
    start_node = Node(None, start)
    start_node.h = heuristic_function(start_node.position, end)
    open_nodes.append(start_node)

    end_node = Node(None, end)

    while open_nodes:
        """
        print("[", end='')
        for n in open_nodes:
            print(str(n)+", ", end='')
        print("]")
        """
        current_node = open_nodes.pop(0)
        closed_nodes.append(current_node)

        # Found goal, re-tracking to find path returns path as list of positions
        if current_node == end_node:
            print("endgame")
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.best_parent
            return path[::-1]  # Returns path in reverse

        # Generate children, they can be north, south, east or west of the parent
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            # For moving diagonally [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # It's a wall, cannot walk here
            if board[node_position[0]][node_position[1]] == 255:
                continue

            # Create new child in this new position
            new_node = Node(current_node, node_position)
            children.append(new_node)

        # Adds the child to the parents child-list
        for child in children:
            # The tile cost is the g cost of the current child, it is found on the board
            tile_cost = board[child.position[0]][child.position[1]]

            # Checks if the child has previously been created, and therefor is either in open or closed nodes
            # if it is, we rather look at the old version and update it
            for i in range(len(open_nodes)):
                if open_nodes[i] == child:
                    child = open_nodes[i]
            for j in range(len(closed_nodes)):
                if closed_nodes[j] == child:
                    child = closed_nodes[j]

            # appending the correct node to children list
            current_node.children.append(child)

            if child not in open_nodes and child not in closed_nodes:
                # It has not yet been evaluated, and we don't need to propagate it
                attach_and_eval(child, current_node, end, tile_cost)
                open_nodes.append(child)
                open_nodes.sort()

            elif current_node.g + tile_cost < child.g:  # (found cheaper path to the child):
                # ∗ attach-and-eval(S,X)
                # ∗ If S ∈ CLOSED then propagate-path-improvements(S)
                attach_and_eval(child, current_node, end, tile_cost)
                if child in closed_nodes:
                    propagate_path_improvements(child)

    return False    # FAIL


def heuristic_function(node, goal):
    """Uses manhattan distance to calculate; it ignores walls and is therefor admissible"""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def attach_and_eval(child, parent, end, tile_cost):
    """Attaches a node to its best parent (so far)
    the child's g'value is computed based on parent, h independently"""
    child.best_parent = parent
    child.g = parent.g + tile_cost
    child.h = heuristic_function(child.position, end)


def propagate_path_improvements(parent):
    """Goes through the children and possibly many other decedents
    If parent is no longer their best parent, the propagation ceases,
    if any child can have parent as its best parent it must be updated
    and propagated further to the children of the children"""
    for child in parent.children:
        if parent.g + 1 < child.g:
            child.best_parent = parent
            child.g = parent.g + 1
            propagate_path_improvements(child)


def draw_path(board, path):
    for t in path:
        board.set_cell_value(t, " Ø ", True)

