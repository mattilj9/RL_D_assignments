import gym
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

np.random.seed(123)

#env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 4

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -1.2, 1.2
y_min, y_max = -0.3, 1.2
x_dot_min, x_dot_max = -2.4, 2.4
y_dot_min, y_dot_max = -2, 2
th_min, th_max = -6.28, 6.28
th_dot_min, th_dot_max = -8, 8
cl_min, cl_max = 0, 1
cr_min, cr_max = 0, 1

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 2222  # TODO: Set the correct value.
initial_q = 0  # T3: Set to 50

# Create discretization grnci
x_grnci = np.linspace(x_min, x_max, discr)
y_grnci = np.linspace(y_min, y_max, discr)
x_dot_grnci = np.linspace(x_dot_min, x_dot_max, discr)
y_dot_grnci = np.linspace(y_dot_min, y_dot_max, discr)
th_grnci = np.linspace(th_min, th_max, discr)
th_dot_grnci = np.linspace(th_dot_min, th_dot_max, discr)
cl_grnci = np.linspace(cl_min, cl_max, 2)
cr_grnci = np.linspace(cr_min, cr_max, 2)

q_grnci = np.zeros((discr, discr, discr, discr, discr, discr, 2, 2, num_of_actions)) + initial_q


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    x = find_nearest(x_grnci, state[0])
    y = find_nearest(y_grnci, state[1])
    x_dot = find_nearest(x_dot_grnci, state[2])
    y_dot = find_nearest(y_dot_grnci, state[3])
    th = find_nearest(th_grnci, state[4])
    th_dot = find_nearest(th_dot_grnci, state[5])
    cl = find_nearest(cl_grnci, state[6])
    cr = find_nearest(cl_grnci, state[7])

    return x, y, x_dot, y_dot, th, th_dot, cl, cr


def get_action(state, q_values, greedy=False):
    # TODO: Implement epsilon-greedy
    cell_index = get_cell_index(state)
    ci1, ci2, ci3, ci4, ci5, ci6, ci7, ci8 = cell_index

    actions = q_values[ci1, ci2, ci3, ci4, ci5, ci6, ci7, ci8, :]
    max_action_index = np.argmax(actions)
    action_index = max_action_index

    if greedy==False:
        # random float in the half-open interval [0.0, 1.0)
        random_sample = np.random.random_sample()
        #epsilon = 0.2  #Task 1.1 - fixed
        epsilon = a / (a + ep)  #Task 1.1 - GLIE        
        #epsilon = 0.0  #Task 1.3
    
        if random_sample < epsilon:         
            action_index = np.random.randint(2)

    return action_index


def update_q_value(old_state, action, new_state, reward, done, q_array):
    # TODO: Implement Q-value update
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)

     # max_action for new state:
    nci1, nci2, nci3, nci4, nci5, nci6, nci7, nci8 = new_cell_index
    actions = q_array[nci1, nci2, nci3, nci4, nci5, nci6, nci7, nci8, :]
    max_a_index = np.argmax(actions)  

    # update old Q (=state-action) values
    oci1, oci2, oci3, oci4, oci5, oci6, oci7, oci8 = old_cell_index

    if done:
        q_array[oci1, oci2, oci3, oci4, oci5, oci6, oci7, oci8, action] = 0 
    else:
        q_array[oci1, oci2, oci3, oci4, oci5, oci6, oci7, oci8, action] = q_array[oci1, oci2, oci3, oci4, oci5, oci6, oci7, oci8, action] + alpha * (reward + gamma * q_array[nci1, nci2, nci3, nci4, nci5, nci6, nci7, nci8, max_a_index] - q_array[oci1, oci2, oci3, oci4, oci5, oci6, oci7, oci8, action]) 


# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = 0.0  # T1: GLIE/constant, T3: Set to 0
    while not done:
        action = get_action(state, q_grnci, greedy=test)
        new_state, reward, done, _ = env.step(action)
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grnci)
        else:
            env.render()
        state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))

# Save the Q-value array
#np.save("q_values.npy", q_grnci)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
#values = np.zeros(q_grnci.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRnci
#values = np.amax(q_grnci, axis=4)

#np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
#average over indexes
#ave_values = np.mean(values, axis=(1, 3))

#sns.heatmap(ave_values, cmap="Blues", annot=True, fmt=".1f")
#plt.title("Heatmap of the value function in terms of x and theta")
#plt.ylabel('Position of the cart [index of x]')
#plt.xlabel('Angle of the pole [index of x theta]')
#plt.figure(2)


# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

