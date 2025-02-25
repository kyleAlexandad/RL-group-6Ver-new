# main.py
import os
import sympy as sp
import torch
import torch.optim as optim
from state import State, CircuitNode, is_terminal
from network import AlphaZeroNet
from trainer import self_play_episode, update_network

# Hyperparameters.
MAX_ACTIONS = 50         # Fixed maximum action space size.
INPUT_SIZE = 2           # Size of the state encoding.
HIDDEN_SIZE = 64         # Hidden layer size.
NUM_SIMULATIONS = 50     # Number of MCTS simulations per move.
MAX_DEPTH = 5            # Maximum number of operations allowed.
NUM_EPISODES = 200       # Number of self-play episodes.
CHECKPOINT_PATH = "checkpoint.pth"  # File to save/load the model.

def main():
    # Prompt the user to enter a polynomial.
    user_input = input("Enter a polynomial in terms of variables (e.g., x**2 + y**2 + 2*x*y): ")
    try:
        target = sp.sympify(user_input)
    except Exception as e:
        print("Error parsing polynomial:", e)
        return
    target = sp.simplify(target)
    print("Target polynomial:", target)
    
    # Extract free symbols from the target.
    free_symbols = list(target.free_symbols)
    if len(free_symbols) == 0:
        print("Target is a constant:", target)
        return
    print("Free symbols in target:", free_symbols)
    
    # If target is simply a single variable, return it immediately.
    if len(target.free_symbols) == 1 and target == free_symbols[0]:
        print("Trivial target detected. Circuit is simply:")
        print(target)
        return

    # Otherwise, build the initial state.
    # For each free symbol, add two copies so that operations (e.g. multiplication for squaring) are possible.
    initial_nodes = []
    for sym in free_symbols:
        initial_nodes.append(CircuitNode(sym))
        initial_nodes.append(CircuitNode(sym))
    # Optionally add constant nodes.
    initial_nodes.append(CircuitNode(1))
    initial_nodes.append(CircuitNode(2))
    initial_nodes.append(CircuitNode(3))
    initial_state = State(initial_nodes, [])
    
    # Create the neural network and optimizer.
    net = AlphaZeroNet(INPUT_SIZE, HIDDEN_SIZE, MAX_ACTIONS)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Load a checkpoint if it exists.
    start_episode = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"] + 1
        print(f"Loaded checkpoint from episode {start_episode}")
    
    # Training loop.
    for episode in range(start_episode, NUM_EPISODES):
        trajectory, reward, final_state = self_play_episode(
            initial_state, target, net, MAX_ACTIONS, NUM_SIMULATIONS, MAX_DEPTH
        )
        loss = update_network(net, optimizer, trajectory, reward)
        print(f"Episode {episode+1}/{NUM_EPISODES}: Reward = {reward:.2f}, Loss = {loss:.2f}")
        if is_terminal(final_state, target):
            print("Found valid circuit in episode", episode+1)
            print("Circuit tree:")
            final_state.nodes[0].print_tree()
            break
        
        # Save the checkpoint after every episode.
        torch.save({
            "episode": episode,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)
    
    # Final test run.
    print("\nFinal test:")
    trajectory, reward, final_state = self_play_episode(
        initial_state, target, net, MAX_ACTIONS, NUM_SIMULATIONS, MAX_DEPTH
    )
    if is_terminal(final_state, target):
        print("Found a valid circuit!")
        final_state.nodes[0].print_tree()
    else:
        print("Did not find a valid circuit.")
    print("Final reward:", reward)

if __name__ == "__main__":
    main()





# # main.py
# import sympy as sp
# import torch
# import torch.optim as optim
# from state import State, CircuitNode, is_terminal
# from network import AlphaZeroNet
# from trainer import self_play_episode, update_network

# # Hyperparameters.
# MAX_ACTIONS = 50         # Fixed maximum action space size.
# INPUT_SIZE = 2           # Size of the state encoding.
# HIDDEN_SIZE = 64         # Hidden layer size.
# NUM_SIMULATIONS = 50     # Number of MCTS simulations per move.
# MAX_DEPTH = 5            # Maximum number of operations allowed.
# NUM_EPISODES = 200       # Number of self-play episodes.

# def main():
#     # Set up the target: x**2 + y**2 + 2*x*y, which simplifies to (x+y)**2.
#     x, y = sp.symbols('x y')
#     target = sp.simplify(x**2 + y**2 + 2*x*y)
#     print("Target polynomial:", target)
    
#     # For (x+y)**2 we need two copies each of x and y.
#     initial_nodes = [CircuitNode(x), CircuitNode(x), CircuitNode(y), CircuitNode(y)]
#     initial_state = State(initial_nodes, [])
    
#     # Create the network and optimizer.
#     net = AlphaZeroNet(INPUT_SIZE, HIDDEN_SIZE, MAX_ACTIONS)
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
    
#     for episode in range(NUM_EPISODES):
#         trajectory, reward, final_state = self_play_episode(initial_state, target, net, MAX_ACTIONS, NUM_SIMULATIONS, MAX_DEPTH)
#         loss = update_network(net, optimizer, trajectory, reward)
#         print(f"Episode {episode+1}/{NUM_EPISODES}: Reward = {reward:.2f}, Loss = {loss:.2f}")
#         if is_terminal(final_state, target):
#             print("Found valid circuit in episode", episode+1)
#             print("Circuit tree:")
#             final_state.nodes[0].print_tree()
#             break
    
#     print("\nFinal test:")
#     trajectory, reward, final_state = self_play_episode(initial_state, target, net, MAX_ACTIONS, NUM_SIMULATIONS, MAX_DEPTH)
#     if is_terminal(final_state, target):
#         print("Found a valid circuit!")
#         final_state.nodes[0].print_tree()
#     else:
#         print("Did not find a valid circuit.")
#     print("Final reward:", reward)

# if __name__ == "__main__":
#     main()
