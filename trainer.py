# trainer.py
import torch
from state import get_possible_actions, is_terminal, compute_reward, apply_action
from network import encode_state
from mcts import run_mcts

def self_play_episode(initial_state, target, net, max_actions, num_simulations, max_depth):
    """
    Runs one self-play episode using MCTS to select moves.
    Returns:
      - trajectory: list of (encoded state, MCTS policy target) pairs.
      - final_reward: computed via compute_reward (heuristic if not terminal).
      - final_state: the resulting state.
    """
    trajectory = []
    state = initial_state
    # Limit the episode length to max_depth moves.
    while not is_terminal(state, target) and len(state.circuit_history) < max_depth:
        valid_actions = get_possible_actions(state)
        if not valid_actions:
            break
        best_action, root = run_mcts(state, target, net, max_actions, num_simulations, max_depth)
        policy = [0] * max_actions
        for action, child in root.children.items():
            if action in valid_actions:
                idx = valid_actions.index(action)
                policy[idx] = child.visit_count
        policy_sum = sum(policy[:len(valid_actions)])
        if policy_sum > 0:
            policy = [x / policy_sum for x in policy[:len(valid_actions)]] + [0] * (max_actions - len(valid_actions))
        else:
            uniform = 1.0 / len(valid_actions)
            policy = [uniform] * len(valid_actions) + [0] * (max_actions - len(valid_actions))
        state_vec = encode_state(state)
        trajectory.append((state_vec, torch.tensor(policy, dtype=torch.float32)))
        state = apply_action(state, best_action)
    reward = compute_reward(state, target)
    return trajectory, reward, state

def update_network(net, optimizer, trajectory, reward, loss_weight=1.0):
    """
    Updates the network parameters using the self-play trajectory.
    Loss combines a cross-entropy (policy) loss and an MSE (value) loss.
    """
    total_policy_loss = 0
    total_value_loss = 0
    for state_vec, target_policy in trajectory:
        policy_logits, value = net(state_vec)
        log_probs = torch.log_softmax(policy_logits, dim=0)
        policy_loss = -torch.sum(target_policy * log_probs)
        value_loss = (reward - value.squeeze()) ** 2
        total_policy_loss += policy_loss
        total_value_loss += value_loss
    loss = total_policy_loss + loss_weight * total_value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
