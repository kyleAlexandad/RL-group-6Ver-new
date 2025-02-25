# state.py
import sympy as sp

class CircuitNode:
    """
    Represents a node in the arithmetic circuit.
    For leaves, operator is None and expr holds a variable or constant.
    For internal nodes, operator is 'add' or 'mul' and left/right point to child nodes.
    """
    def __init__(self, expr, operator=None, left=None, right=None):
        self.expr = sp.simplify(expr)  # Simplify the expression.
        self.operator = operator       # 'add' or 'mul' for internal nodes; None for leaves.
        self.left = left               # Left child (CircuitNode) if internal.
        self.right = right             # Right child (CircuitNode) if internal.
    
    def is_leaf(self):
        return self.operator is None
    
    def __str__(self):
        if self.is_leaf():
            return str(self.expr)
        else:
            return f"({str(self.left)} {self.operator} {str(self.right)})"
    
    def print_tree(self, indent=0):
        """Recursively prints the circuit tree."""
        space = " " * indent
        if self.is_leaf():
            print(space + str(self.expr))
        else:
            print(space + self.operator.upper() + " gate")
            self.left.print_tree(indent + 4)
            self.right.print_tree(indent + 4)

class State:
    """
    Represents the current state of the synthesis game.
    Attributes:
      - nodes: list of CircuitNode objects (available subexpressions).
      - circuit_history: list of actions applied to reach this state.
    """
    def __init__(self, nodes, circuit_history):
        self.nodes = nodes  # List of CircuitNode objects.
        self.circuit_history = circuit_history

def get_possible_actions(state):
    """
    Returns a list of valid actions.
    Each action is a tuple (op, i, j) where:
      - op is 'add' or 'mul'
      - i, j are indices in state.nodes.
    """
    actions = []
    n = len(state.nodes)
    if n < 2:
        return actions
    for i in range(n):
        for j in range(i + 1, n):
            actions.append(('add', i, j))
            actions.append(('mul', i, j))
    return actions

def apply_action(state, action):
    """
    Applies the given action to the state and returns a new state.
    Combines two nodes using the specified operation and creates a new CircuitNode.
    """
    op, i, j = action
    node_a = state.nodes[i]
    node_b = state.nodes[j]
    if op == 'add':
        new_expr = sp.simplify(node_a.expr + node_b.expr)
    elif op == 'mul':
        new_expr = sp.simplify(node_a.expr * node_b.expr)
    else:
        raise ValueError("Unknown operation")
    new_node = CircuitNode(new_expr, operator=op, left=node_a, right=node_b)
    new_nodes = state.nodes.copy()
    for index in sorted([i, j], reverse=True):
        new_nodes.pop(index)
    new_nodes.append(new_node)
    new_history = state.circuit_history + [action]
    return State(new_nodes, new_history)

def is_terminal(state, target):
    """
    A state is terminal if there is exactly one node and its expression is (symbolically) equal to target.
    """
    if len(state.nodes) == 1:
        return sp.simplify(state.nodes[0].expr - target) == 0
    return False

def compute_reward(state, target):
    """
    Computes a reward for the state.
    - If terminal, reward = 100 minus the number of operations (circuit depth).
    - Otherwise, a heuristic reward is given based on the similarity between one of the state's
      node expressions and the target.
      
    The similarity is defined here as:
        similarity = 1 / (1 + |len(str(expr)) - len(str(target))| )
    This is a very rough measure but provides some gradient.
    """
    if is_terminal(state, target):
        return 100 - len(state.circuit_history)
    # Otherwise, try to find a node that is "close" to the target.
    best_sim = 0
    for node in state.nodes:
        # If any node already equals target, that's perfect.
        if sp.simplify(node.expr - target) == 0:
            return 100 - len(state.circuit_history)
        diff = abs(len(str(node.expr)) - len(str(target)))
        sim = 1.0 / (1.0 + diff)
        if sim > best_sim:
            best_sim = sim
    # Scale reward by the similarity factor.
    return (100 - len(state.circuit_history)) * best_sim
