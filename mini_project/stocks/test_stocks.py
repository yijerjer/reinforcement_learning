import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stock_env import StockEnv
from actor_critic_eligibility_trace import ActorCriticEligibilityTrace

test = True

env = StockEnv(test=test, random_flips=False)
env_passive = StockEnv(test=test)
ac = ActorCriticEligibilityTrace(env)

if not test:
    ac.train()
else:
    df = pd.read_csv("csvs/norm_all_stocks_5yr.csv")
    stock_df = df[df.Name == env.test_stock_name]

    policy_mlp = ac.policy_mlp
    policy_mlp.load_state_dict(torch.load("policy_mlp.pth"))
    value_mlp = ac.value_mlp
    value_mlp.load_state_dict(torch.load("value_mlp.pth"))

    obss = []
    actions = []
    rewards = []

    obs = env.reset()
    while True:
        obss.append(obs)
        action, _ = policy_mlp(
            torch.as_tensor(obs, dtype=torch.float32)
        )
        obs, reward, done, _ = env.step(action.detach().numpy())
        actions.append(action)
        rewards.append(reward)

        if done:
            break

    obss_passive = []
    actions_passive = []
    rewards_passive = []

    obs = env.reset()
    while True:
        obss_passive.append(obs)
        obs, reward, done, _ = env.step(1)
        actions_passive.append(action)
        rewards_passive.append(reward)

        if done:
            break

    np_obss = np.array(obss)
    closes = np_obss[:, 3]
    stock_mean = stock_df.close.mean()
    stock_std = stock_df.close.std()
    original_closes = closes * stock_std + stock_mean

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(original_closes, label="Price")

    buys = []
    sells = []
    for i in range(1, len(actions)):
        if actions[i] == 1 and actions[i-1] == 0:
            buys.append((i, original_closes[i]))
        if actions[i] == 0 and actions[i-1] == 1:
            sells.append((i, original_closes[i]))
    buys = np.array(buys)
    sells = np.array(sells)

    plt.scatter(buys[:, 0], buys[:, 1], color="g", label="Buy", s=10)
    plt.scatter(sells[:, 0], sells[:, 1], color="r", label="Sell", s=10)

    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.legend()
    plt.title(f"Stock Price of {env.test_stock_name}")

    plt.subplot(2, 1, 2)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.plot([sum(rewards[0:i]) for i, _ in enumerate(rewards)], 
             label="With RL")
    plt.plot(
        [sum(rewards_passive[0:i]) for i, _ in enumerate(rewards_passive)],
        label="Passive Investing"
    )
    plt.ylabel("Returns")
    plt.xlabel("Time")
    plt.legend()

    plt.savefig(f"plots/{env.test_stock_name}_plot.png", dpi=300)
    # plt.show()
