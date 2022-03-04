from collections import defaultdict, namedtuple
from random import choice, random, seed
from typing import Hashable
import gym
from numpy import Inf

Action = int
Reward = float
Obs = Hashable
Step = tuple[Obs, Action, Reward]
Episode = list[Step]
episode_step = namedtuple('EpisodeStep', field_names=['obs', 'action', 'reward'])
named_obs_action = namedtuple('obs_action', field_names=['obs', 'action'])

# set random seed
seed(21)


class Average:

    def __init__(self):
        self.n = 0
        self.sum = 0
        self.mean = 0

    def add_value(self, val):
        self.n += 1
        self.sum += val
        self.mean = self.sum / self.n

    def get_mean(self):
        return self.mean


class EpsilonGreedyDiscretePolicy:
    """Epsilon-Greedy Policy for a Discrete Action Space."""

    def __init__(self, env: gym.core.Env):
        self.actions = frozenset(range(env.action_space.n))
        self.n_actions = env.action_space.n

        # Initialize arbitrary policy
        self.policy = defaultdict(int)

    def optimal_action(self, policy_state: Hashable) -> Action:
        return self.policy[policy_state]

    def sample_action(self, policy_state: Hashable, epsilon) -> Action:
        """
        Given a policy_state, returns the optimal action with probability: 1 - eps + (eps / |A(S_t|).
        And returns non-optimal actions with probability: eps / |A(S_t)|
        """
        prob_optimal = 1 - epsilon + epsilon / self.n_actions
        if random() < prob_optimal:
            return self.policy[policy_state]
        else:
            non_optimal = list(self.actions - {self.policy[policy_state]})
            return choice(non_optimal)

    def set_optimal_action(self, policy_state: Hashable, action: Action) -> None:
        """
        Given a policy_state, sets the optimal action for that policy_state.
        """
        assert action in self.actions
        self.policy[policy_state] = action


def on_policy_first_visit_mc_control(env: gym.core.Env, policy: EpsilonGreedyDiscretePolicy, n: int,
                                     lambda_: float = 0.99, accuracy: float = 0.001, start_epsilon=1):
    # Sutton, Barto. 2018. Reinforcement Learning an Introduction p. 101
    def generate_episode(episode_env: gym.core.Env, ep_policy: EpsilonGreedyDiscretePolicy, epsilon) -> Episode:
        episode_state = episode_env.reset()
        is_done = False
        episode = []

        while not is_done:
            action = ep_policy.sample_action(episode_state, epsilon)
            new_state, reward, is_done, _ = episode_env.step(action)
            step = episode_step(episode_state, action, reward)
            episode.append(step)
            episode_state = new_state
        return episode

    # Initialize Q-Values
    q = defaultdict(float)

    # Holds the average return for each state-action pair. Key: (state, action). Values: Average object
    returns = defaultdict(Average)

    error = Inf
    while n > 0:
        print(n, "episodes left")
        print("current error is ", error)
        error = 0

        # generate an episode
        curr_epsilon = start_epsilon - (1/n)
        episode = generate_episode(env, policy, curr_epsilon)
        print("episode reward: ", episode[-1].reward)
        print("episode length: ", len(episode))
        # initialize episode return to 0
        ep_return = 0
        # Loop over episode steps in reverse, update q-values, and update the policy
        for idx, curr_step in enumerate(reversed(episode)):
            # Timestep = 0, 1, ..., T
            timestep = len(episode) - idx - 1
            # The return for the current step is the current reward plus the discounted future returns
            ep_return = curr_step.reward + lambda_ * ep_return

            # Get state actions that occurred before the current timestep
            prior_steps = episode[:timestep]
            prior_state_actions = [(step.obs, step.action) for step in prior_steps]

            # This is first-visit Monte Carlo control. Only count the first occurrence of a state-action pair in an
            # episode
            if (curr_step.obs, curr_step.action) not in prior_state_actions:
                # Update the average return for the current (state, action) pair
                returns[(curr_step.obs, curr_step.action)].add_value(ep_return)

                # Update the q-value to be an average of historical returns
                new_q_value = returns[(curr_step.obs, curr_step.action)].get_mean()
                q_error = abs(new_q_value - q[(curr_step.obs, curr_step.action)])
                if q_error > error:
                    error = q_error
                q[curr_step.obs, curr_step.action] = new_q_value

                # Find the optimal action for the current state and update the policy
                best_action = 0
                best_action_value = 0
                for (q_state, q_action), q_value in q.items():
                    if q_state == curr_step.obs and q_value > best_action_value:
                        best_action_value = q_value
                        best_action = q_action
                policy.set_optimal_action(curr_step.obs, best_action)
        n -= 1
    return policy, q

if __name__ == "__main__":
    from tabular_monte_carlo import on_policy_first_visit_mc_control, EpsilonGreedyDiscretePolicy
    from grid_env import ImmutableObjectDirectionWrapper
    from gym_minigrid.wrappers import FullyObsWrapper
    import matplotlib.pyplot as plt


    env = ImmutableObjectDirectionWrapper(FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0")))
    obs = env.reset()
    policy = EpsilonGreedyDiscretePolicy(env)
    policy, q = on_policy_first_visit_mc_control(env, policy, 1000, lambda_=1)

# TODO - Double check in the code
    # In the Q dict, am I saving the state correct? Is the key correct?
# TODO - Copy someone elses code. See how many steps it takes for the other code to solve this
# TODO - figure out why this was a bug before: prior_steps = episode[:timestep]
# TODO - Add in tensorboard for logging instead of print statements
# TODO - read over code and clean it up a bit. Especially the policy code.
# TODO - read over Andy Jones article. See if there are tips that you can quickly implement. Don't spend a ton of time implementing stuff there