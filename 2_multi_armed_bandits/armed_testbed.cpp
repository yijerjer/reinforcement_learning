#include <iostream>
#include <fstream>
#include <tuple>
#include <random>
#include <algorithm>
using namespace std;


class SampleDistribution {
    static random_device rd;
    static mt19937 generator;

    public:
    static double uniform_real(double min = 0, double max = 1) {
        uniform_real_distribution<double> uni_dist(min, max);
        return uni_dist(generator);
    }

    static int uniform_int(int max) {
        uniform_int_distribution<int> uni_dist(0, max);
        return uni_dist(generator);
    }

    static double normal(double mean = 0, double variance = 1) {
        normal_distribution<double> norm_dist(mean, variance);
        return norm_dist(generator);
    }

};
random_device SampleDistribution::rd;
mt19937 SampleDistribution::generator(SampleDistribution::rd());


class KArmedTestbed {
    protected:
    bool has_rw;
    
    public:
    vector<double> arm_values;
    int K;
    KArmedTestbed() {}
    KArmedTestbed(int K_arms, bool has_random_walk = false) {
        K = K_arms;
        has_rw = has_random_walk;
        if (has_rw) {
            arm_values.resize(K_arms, 0);
        } else {
            arm_values.resize(K_arms);
            generate(arm_values.begin(), arm_values.end(), []() { return SampleDistribution::normal(0); });
        }
    }
    ~KArmedTestbed() {}

    double get_Kth_reward(int Kth_arm) {
        double mean = arm_values[Kth_arm];
        if (has_rw) {
            update();
        }
        return SampleDistribution::normal(mean, 1);
    }

    int is_Kth_optimal(int Kth_arm) {
        int max_arm = max_element(arm_values.begin(), arm_values.end()) - arm_values.begin();
        return Kth_arm == max_arm ? 1 : 0;
    }

    void update() {
        vector<double> random_walk(K);
        generate(random_walk.begin(), random_walk.end(), []() { return SampleDistribution::normal(0, 0.01); });
        transform(arm_values.begin(), arm_values.end(), random_walk.begin(), arm_values.begin(), plus<double>());
    }
};


class ActionValueAgent {
    KArmedTestbed testbed;
    double epsilon;
    int timesteps;
    bool fixed_step;
    vector<int> Kth_occurence;
    vector<double> Q_values;

    public:
    ActionValueAgent(KArmedTestbed ka_testbed, double explore_epsilon, int n_timesteps, bool fixed_step_param = false) {
        testbed = ka_testbed;
        epsilon = explore_epsilon;
        timesteps = n_timesteps;
        fixed_step = fixed_step_param;
        Kth_occurence.resize(testbed.K, 0);
        Q_values.resize(testbed.K, 0);
    }

    vector<tuple<double, int>> single_run() {
        int print_every = 50;
        vector<tuple<double, int>> reward_at_t(timesteps);
        vector<int> optimal_at_t(timesteps, 0);
        for (int t = 0; t < timesteps; t++) {
            int next_act = pick_next_action();
            Kth_occurence[next_act] += 1;
            double reward = testbed.get_Kth_reward(next_act);
            tuple<double, int> tup = make_tuple(reward, testbed.is_Kth_optimal(next_act));
            reward_at_t[t] = tup;

            update_Q_estimates(next_act, reward);
        }
        return reward_at_t;
    }

    int pick_next_action() {
        vector<double> Q_estimates = Q_values;
        double max_Q = *max_element(Q_estimates.begin(), Q_estimates.end());
        vector<double> max_Q_idxs;
        for (int i = 0; i < testbed.K; i++) {
            if (Q_estimates[i] == max_Q) {
                max_Q_idxs.push_back(i);
            }
        }
        
        int next_act;
        
        if (should_explore()) {
            next_act = SampleDistribution::uniform_int(testbed.K - 1);
        } else {
            if (max_Q_idxs.size() == 1) {
                next_act = max_Q_idxs[0];
            } else {
                next_act = max_Q_idxs[SampleDistribution::uniform_int(max_Q_idxs.size() - 1)];
            }
        }
        return next_act;
    }

    bool should_explore() {
        double rand_val = SampleDistribution::uniform_real();
        return rand_val < epsilon ? true : false;
    }

    void update_Q_estimates(int action, double reward) {
        double step_size = fixed_step ? 0.1 : (1.0 / Kth_occurence[action]);  
        Q_values[action] += step_size * (reward - Q_values[action]);
    }
};

void rewards_to_csv(vector<vector<tuple<double, double>>> all_rewards, string filename) {
    printf("Writing results to file... \n");
    int timesteps = all_rewards[0].size();
    ofstream csv_file(filename);
    int n_rows = all_rewards.size();
    for (int t = 0; t < timesteps; t++) {
        for (int i = 0; i < n_rows; i++) {
            csv_file << get<0>(all_rewards[i][t]) << ", ";
            string end_line = (i == n_rows - 1) ? "\n" : ", "; 
            csv_file << get<1>(all_rewards[i][t]) << end_line;
        }
    }
    csv_file.close();
}

void rewards_average_to_csv(vector<vector<tuple<double, double>>> all_rewards, string filename, vector<double> epsilons) {
    int timesteps = all_rewards[0].size();
    int half = timesteps / 2;
    printf("Calculating average from final %d steps... \n", half);
    vector<double> reward_sums(all_rewards.size(), 0);

    for (int t = half; t < timesteps; t++) {
        for (int i = 0; i < all_rewards.size(); i++) {
            reward_sums[i] += get<0>(all_rewards[i][t]);
        }
    }

    printf("Writing results to file... \n");
    ofstream csv_file(filename);
    for (int i = 0; i < reward_sums.size(); i++) {
        csv_file << epsilons[i] << ", " <<  reward_sums[i] / half << endl;
    }    
}


int main() {
    int n_runs = 2000;
    int timesteps = 1000;
    int print_every = 200;
    vector<double> epsilons { 0, 0.01, 0.1 };
    vector<vector<tuple<double, double>>> all_reward_average;

    for (double epsilon : epsilons) {
        printf("EPSILON = %.5f \n", epsilon);
        vector<tuple<double, int>> reward_at_t_sum(timesteps, make_tuple(0, 0));
        for (int i = 1; i <= n_runs; i++) {
            KArmedTestbed testbed = KArmedTestbed(10, true);
            ActionValueAgent agent = ActionValueAgent(testbed, epsilon, timesteps, true);
            vector<tuple<double, int>> reward_at_t = agent.single_run();

            for (int t = 0; t < timesteps; t++) {
                get<0>(reward_at_t_sum[t]) += get<0>(reward_at_t[t]);
                get<1>(reward_at_t_sum[t]) += get<1>(reward_at_t[t]);
            }  
            if (i % print_every == 0) {
                printf("Run %d/%d \n", i, n_runs);
            }
        }
        vector<tuple<double, double>> reward_at_t_ave(timesteps);
        transform(reward_at_t_sum.begin(), reward_at_t_sum.end(), reward_at_t_ave.begin(),
            [n_runs](auto tup) { return make_tuple(get<0>(tup) / (double) n_runs, get<1>(tup) / (double) n_runs); });
        all_reward_average.push_back(reward_at_t_ave);
    }

    rewards_to_csv(all_reward_average, "csvs/nonstationary_fixed_step.csv");
    // rewards_average_to_csv(all_reward_average, "csvs/parameter_fixed_step.csv", epsilons);
}