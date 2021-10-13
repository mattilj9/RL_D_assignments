# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld

epsilon = 10e-4  # TODO: Use this criteria for Task 3

# Set up the environment
env = SailingGridworld(rock_penalty=-2) 
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)


if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    #value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))

    # Task 1
    #value_est, policy = env.value_iteration_task1(value_est, policy, niters=100, gamma=0.9)

    # Task 3
    #value_est, policy = env.value_iteration_task3(value_est, policy, gamma=0.9, epsilon = 10e-4)

    # Show the values and the policy
    #env.draw_values(value_est)
    #env.draw_actions(policy)
    #env.render()
    #sleep(1)

    # Save the state values and the policy
    #fnames = "values.npy", "policy.npy"
    #np.save(fnames[0], value_est)
    #np.save(fnames[1], policy)
    #print("Saved state values and policy to", *fnames)

    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    N = 1000  #number of episodes
    gamma = 0.9
    rewards = np.zeros(N) # store rewards of N episodes

    for iter in np.arange(N):
        if ((iter+1)%100==0): print("{} ".format(iter+1), sep=' ', end=' ', flush=True)
        env.reset()
        t = 0  #time step        
        Gt = 0
        done = False
        while not done:
            # Select a random action
            # TODO: Use the policy to take the optimal action (Task 2)
            #x, y = state
            #action = policy[x, y]
            action = int(np.random.random()*4)

            # Step the environment
            state, reward, done, _ = env.step(action)

            Gt = Gt + np.power(gamma, t) * reward
            t = t + 1
            
            # Render and sleep
            #env.render()
            #sleep(0.5)

        rewards[iter] = Gt

    print("\nDiscounted reward: mean= {}, std= {}".format(np.mean(rewards), np.std(rewards)))