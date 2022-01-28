1. Consider a k-armed bandit problem with k = 4 actions, denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using ε-greedy action selection, sample-average action-value estimates, and initial estimates of Q1(a) = 0, for all a. Suppose the initial sequence of actions and rewards is A1 = 1, R1 = −1, A2 = 2, R2 = 1, A3 = 2, R3 = −2, A4 = 2, R4 = 2, A5 = 3, R5 = 0. On some of these time steps the ε case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?

* a T1 → A1 = 1, R1 = −1 → A1 = -1, A2 = 0, A3 = 0, A4 = 0
* b T2 → A2 = 2, R2 = 1 → A1 = -1, A2 = 1, A3 = 0, A4 = 0
* c T3 → A3 = 2, R3 = −2 → A1 = -1, A2 = -1/2, A3 = 0, A4 = 0
* d T4 → A4 = 2, R4 = 2 → A1 = 1, A2 = 1/3, A3 = 0, A4 = 0 (definitely occured)
* e T5 → A5 = 3, R5 = 0 → A1 = 1, A2 = 1/3, A3 = 0, A4 = 0 (definitely occured)

Timesteps four and five definitely had the epsilon case occur since it chose an action with a sample-average action-value estimate lower than the maximum. In terms of time steps that epsilon possibly could have occured, all time steps are possible, because random choice includes the maximum sample-average action-value choice, and thus even moves that look opitmal may have been random.

