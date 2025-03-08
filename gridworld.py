import numpy as np

def grid_world_value_iteration(grid_size=4, gamma=1, theta=1e-4):
    """
    Performs value iteration on a 4x4 GridWorld.

    Args:
        grid_size: Size of the GridWorld (NxN).
        gamma: Discount factor.
        theta: Convergence threshold.

    Returns:
        A NumPy array representing the converged value function.
    """

    num_states = grid_size * grid_size
    V = np.zeros(num_states)
    rewards = np.full(num_states, -1)
    rewards[-1] = 0  # Terminal state reward

    def state_to_row_col(state):
        return state // grid_size, state % grid_size

    def row_col_to_state(row, col):
        return row * grid_size + col

    def get_next_state(state, action):
        row, col = state_to_row_col(state)
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(grid_size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(grid_size - 1, col + 1)
        return row_col_to_state(row, col)

    actions = [0, 1, 2, 3]  # Up, Down, Left, Right
    transition_prob = 0.25  # Equal probability for each action

    while True:
        delta = 0
        V_new = V.copy()
        for s in range(num_states - 1):  # Exclude terminal state
            v = V[s]
            expected_values = []
            for a in actions:
                s_prime = get_next_state(s, a)
                expected_values.append(rewards[s] + gamma * V[s_prime])

            V_new[s] = transition_prob * sum(expected_values)

            delta = max(delta, abs(V_new[s] - v))

        V = V_new
        if delta < theta:
            break

    return V.reshape((grid_size, grid_size))

# Run value iteration
final_value_function = grid_world_value_iteration()
print(final_value_function)