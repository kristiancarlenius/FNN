import numpy as np


def func(X: np.ndarray):
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2

def noisy_func(X: np.ndarray, epsilon: float = 0.075):
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon

def get_data(n_train: int, n_test: int):
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)
    return X_train, y_train, X_test, y_test

def perceptron(x, w, b):
    return np.dot(w, x) + b.reshape(-1, 1)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def passive_td_learner(percept, pi, U, N_s, s):
    """
    Passive-TD-Learner algorithm for reinforcement learning.
    """
    s_prime, r = percept  # unpack the percept into current state and reward

    if s_prime not in U:  # if current state is new, set its utility to 0
        U[s_prime] = 0

    if s is not None:  # if previous state is not null
        N_s[s] += 1  # increment the frequency count for previous state
        alpha = 1.0 / N_s[s]  # compute learning rate based on frequency count
        U[s] += alpha * (r + U[s_prime] - U[s])  # update utility for previous state

    s = s_prime  # update the previous state to current state

    return pi[s_prime]  # return the action for the current state based on the fixed policy


def passive_adp_learner(percept, pi, mdp, U, N, s, a):
    """
    Passive-ADP-Learner algorithm for reinforcement learning.
    """
    s_prime, r = percept  # unpack the percept into current state and reward

    if s_prime not in U:  # if current state is new, set its utility to 0
        U[s_prime] = 0

    if s is not None:  # if previous state and action are not null
        N[s, a, s_prime] += 1  # increment the outcome count for previous state and action
        mdp.R[s, a, s_prime] = r  # set the reward for the previous state and action
        if a not in mdp.A[s]:  # if the action is new for the previous state, add it to the list of actions
            mdp.A[s].append(a)
        P = np.zeros((len(mdp.S), len(mdp.A), len(mdp.S)))  # initialize the transition probabilities
        for s_idx in range(len(mdp.S)):  # compute the transition probabilities for each state-action pair
            for a_idx in range(len(mdp.A[s_idx])):
                a = mdp.A[s_idx][a_idx]
                N_s = N[s_idx, a_idx, :]
                P[s_idx, a_idx, :] = N_s / np.sum(N_s)
        mdp.P = P  # set the transition probabilities for the MDP
        U = policy_evaluation(pi, U, mdp)  # evaluate the policy using the updated MDP and utilities

    s, a = s_prime, pi[s_prime]  # update the previous state and action to the current state and policy action

    return a


def policy_evaluation(pi, U, mdp):
    """
    Policy Evaluation algorithm for fixed-policy Bellman equations.
    """
    theta = 0.01  # convergence threshold
    while True:
        delta = 0  # track the maximum change in utility
        for s_idx in range(len(mdp.S)):
            for a_idx in range(len(mdp.A[s_idx])):
                a = mdp.A[s_idx][a_idx]
                Q = 0
                for s_prime_idx in range(len(mdp.S)):
                    Q += mdp.P[s_idx, a_idx, s_prime_idx] * (mdp.R[s_idx, a_idx, s_prime_idx] + mdp.discount_factor * U[s_prime_idx])
                delta = max(delta, abs(U[s_idx] - Q))
                U[s_idx] = Q
        if delta < theta:
            break

    return U

if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    """
    E_train = []
    for x in range(len(X_train)):
        if(X_train[x][0]*y_train[x]>=0):
            o = 1
        else:
            o = -1
        E_train.append((1/2)*((X_train[x][1]-o)^2))
    """
    #pi er sigmoid
    # TODO: Your code goes here.
    # Feedforward neural network with one hidden layer that supports regression.
    # Two input features, two units in the hidden layer with sigmoid activation function, and one output unit.
    # Initialize weights and biases
    hidden_weights = np.random.randn(2, 2)
    hidden_bias = np.zeros((2, 1))
    output_weights = np.random.randn(2, 32)
    output_bias = np.zeros((1, 1))
    learning_rate = 0.1
    n_epochs = 1000
    batch_size = 32
    n_batches = len(X_train) 
    # Training loop
    for i in range(n_epochs):
        for j in range(n_batches):
            X_batch = X_train[j * batch_size : (j + 1) * batch_size, :]
            y_batch = y_train[j * batch_size : (j + 1) * batch_size].reshape(-1, 1)

            # Forward pass
            hidden_output = sigmoid(perceptron(X_batch.T, hidden_weights.T, hidden_bias))
            output = perceptron(hidden_output.T, output_weights.T.T, output_bias)

            # Backward pass
            output_error = output - y_batch
            hidden_error = np.dot(output_weights, output_error.T) * hidden_output * (1 - hidden_output)
            output_weights -= learning_rate * np.dot(hidden_output, output_error)
            output_bias -= learning_rate * np.sum(output_error, keepdims=True)
            hidden_weights -= learning_rate * np.dot(X_batch.T, hidden_error.T)
            hidden_bias -= learning_rate * np.sum(hidden_error, axis=1, keepdims=True)

        # Compute and print loss on training and test sets
        train_loss = np.mean((y_train - perceptron(sigmoid(perceptron(X_train.T, hidden_weights.T, hidden_bias)).T, output_weights.T, output_bias)) ** 2)
        test_loss = np.mean((y_test - perceptron(sigmoid(perceptron(X_test.T, hidden_weights.T, hidden_bias)).T, output_weights.T, output_bias)) ** 2)
        print(f"Epoch {i+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")