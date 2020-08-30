import gym
# from policy_gradient import PolicyGradient
# from policy_gradient_baseline import PolicyGradientBaseline
# from actor_critic import ActorCritic
from actor_critic_eligibility_trace import ActorCriticEligibilityTrace


env = gym.make("CartPole-v1")
pg = ActorCriticEligibilityTrace(
    env, render=True, policy_hidden_size=[32], value_hidden_size=[32]
)
pg.train()
