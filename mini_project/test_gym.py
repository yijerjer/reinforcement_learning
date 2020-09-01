import gym
# from algorithms.policy_gradient import PolicyGradient
# from algorithms.policy_gradient_baseline import PolicyGradientBaseline
# from algorithms.actor_critic import ActorCritic
from algorithms.actor_critic_eligibility_trace import ActorCriticEligibilityTrace
# from algorithms.actor_critic_cont import ActorCriticContinuous


env = gym.make("AirRaid-v0")
pg = ActorCriticEligibilityTrace(
    env, render=True, policy_hidden_size=[64], value_hidden_size=[64]
)
pg.train()
