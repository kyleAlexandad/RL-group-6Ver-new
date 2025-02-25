# solver.py
from collections import deque
import sympy as sp
from state import State, CircuitNode, get_possible_actions, apply_action, is_terminal

def bfs_solve(initial_state, target, max_depth=3):
    """
    Performs a breadth-first search over the state space up to a maximum depth.
    Returns a tuple (state, depth) if a terminal state is found (i.e. one node whose expression 
    is symbolically equal to the target). Otherwise, returns None.
    """
    queue = deque()
    queue.append((initial_state, 0))
    visited = set()
    
    while queue:
        state, depth = queue.popleft()
        # Check if this state is terminal.
        if is_terminal(state, target):
            return state, depth
        if depth >= max_depth:
            continue
        actions = get_possible_actions(state)
        for action in actions:
            new_state = apply_action(state, action)
            # Use a simple representation to avoid revisiting equivalent states.
            # We represent the state as a sorted tuple of the string forms of each node’s expression.
            state_repr = tuple(sorted(str(node.expr) for node in new_state.nodes))
            if state_repr in visited:
                continue
            visited.add(state_repr)
            queue.append((new_state, depth + 1))
    return None

if __name__ == "__main__":
    # For example, let the target be: x**2 + y**2 + 2*x*y which simplifies to (x+y)**2.
    x, y = sp.symbols('x y')
    target = sp.simplify(x**2 + y**2 + 2*x*y)
    print("Target polynomial:", target)
    
    # To build the circuit (x+y)**2, we need two copies of x and two copies of y.
    initial_nodes = [CircuitNode(x), CircuitNode(x), CircuitNode(y), CircuitNode(y)]
    initial_state = State(initial_nodes, [])
    
    # Set a depth limit (here, 3 operations should be sufficient).
    result = bfs_solve(initial_state, target, max_depth=3)
    
    if result is not None:
        solved_state, depth = result
        print("Solution found at depth", depth)
        if len(solved_state.nodes) == 1:
            print("Circuit tree:")
            solved_state.nodes[0].print_tree()
        else:
            print("Solution state (multiple nodes):")
            for node in solved_state.nodes:
                print(node)
    else:
        print("No solution found within depth limit.")
