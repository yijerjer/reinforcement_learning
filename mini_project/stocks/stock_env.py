import pandas as pd
import numpy as np


class StockEnv:
    def __init__(self, test=False, test_stock=None, random_flips=True):
        self.df = pd.read_csv("csvs/norm_all_stocks_5yr.csv")
        self.df.date = pd.to_datetime(self.df.date, format="%Y-%m-%d")
        self.names = self.df.Name.unique()
        self.obs_space_size = 5
        self.action_space_size = 2
        self.max_date = pd.to_datetime("2017-02-01", format="%Y-%m-%d")
        self.random_flips = random_flips
        self.test = test

        if test:
            self.test_stock_name = (test_stock if test_stock
                          else np.random.choice(self.names))
            self.curr_stock_df = self.df[
                (self.df.Name == self.test_stock_name) & (self.df.date > self.max_date)
            ]
        else:
            self.curr_stock_df = None
        self.iter = 0
        self.curr_action = 0    # 1 for hold, 0 for no stocks held

        # self.normalise_each_stock()

    def reset(self):
        if not self.test:
            self.curr_stock_df = self.random_stock_period()
        self.iter = 0

        # removes the bias of the bullish or bearish markets by randomly
        # flipping the normalised stock prices
        self.flip = -1 if (self.random_flips and np.random.randint(2)) else 1
        return self.get_obs()

    def step(self, action):
        prev_close = self.curr_stock_df.iloc[self.iter]["close"].copy()
        reward = 0

        self.iter += 1
        done = True if self.iter == (len(self.curr_stock_df) - 1) else False
        if self.curr_action:
            close = self.curr_stock_df.iloc[self.iter]["close"]
            reward = close - prev_close
        self.curr_action = action
        # print(action, reward)
        return self.get_obs(), self.flip*reward, done, dict()

    def get_obs(self):
        return self.flip * np.array(self.curr_stock_df.iloc[self.iter][9:14],
                                    dtype=np.float32)

    def random_stock_period(self, days=60):
        stock_name = np.random.choice(self.names)
        stock_data = self.df[(self.df.Name == stock_name)
                             & (self.df.date < self.max_date)]
        n_data = stock_data.shape[0]
        if n_data > 60:
            rand_start_idx = np.random.randint(n_data - days)
            return stock_data[rand_start_idx:(rand_start_idx + 90)]
        else:
            return stock_data

    def normalise_each_stock(self):
        for header in ["open", "high", "low", "close", "volume"]:
            norm_header = "norm_" + header
            print(header)
            for name in self.names:
                print(name)
                df_stock = self.df[self.df.Name == name]
                if df_stock.shape[0] < 1000:
                    self.df = self.df.drop(df_stock.index)
                mean = df_stock[header].mean()
                std = df_stock[header].std()
                self.df.loc[self.df.Name == name, norm_header] = (
                    (self.df.loc[self.df.Name == name, header] - mean) / std
                )

        self.df.to_csv("csvs/norm_all_stocks_5yr.csv")

