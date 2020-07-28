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
        uniform_int_distribution<int> uni_dist(0, max - 1);
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
    vector<double> arm_values;
    
    public:
    int K;
    KArmedTestbed() {}
    KArmedTestbed(int K_arms) {
        K = K_arms;
        arm_values.resize(K_arms);
        generate(arm_values.begin(), arm_values.end(), []() { return SampleDistribution::normal(0); });
    }
    ~KArmedTestbed() {}

    double get_Kth_reward(int Kth_arm) {
        double mean = arm_values[Kth_arm];
        return SampleDistribution::normal(mean, 1);
    }

    int is_Kth_optimal(int Kth_arm) {
        int max_arm = max_element(arm_values.begin(), arm_values.end()) - arm_values.begin();
        return Kth_arm == max_arm ? 1 : 0;
    }
};


class ActionValueAgent {
    KArmedTestbed testbed;
    double epsilon;
    vector<double> total_reward;
    vector<int> Kth_occurence;

    public:
    ActionValueAgent(KArmedTestbed ka_testbed, double explore_epsilon) {
        testbed = ka_testbed;
        epsilon = explore_epsilon;
        total_reward.resize(testbed.K, 0);
        Kth_occurence.resize(testbed.K, 0);
    }

    vector<tuple<double, int>> single_run() {
        int timesteps = 1000;
        int print_every = 50;
        vector<tuple<double, int>> reward_at_t(timesteps);
        vector<int> optimal_at_t(timesteps, 0);
        for (int t = 0; t < timesteps; t++) {
            int next_act = pick_next_action();
            double reward = SampleDistribution::normal(testbed.get_Kth_reward(next_act));
            total_reward[next_act] += reward;
            Kth_occurence[next_act] += 1;
            tuple<double, int> tup = make_tuple(reward, testbed.is_Kth_optimal(next_act));
            reward_at_t[t] = tup;
        }
        return reward_at_t;
    }

    int pick_next_action() {
        vector<double> Q_estimates = sample_averages();
        double max_Q = *max_element(Q_estimates.begin(), Q_estimates.end());
        vector<double> max_Q_idxs;
        for (int i = 0; i < testbed.K; i++) {
            if (Q_estimates[i] == max_Q) {
                max_Q_idxs.push_back(i);
            }
        }
        
        int next_act;
        
        if (should_explore()) {
            next_act = SampleDistribution::uniform_int(testbed.K);
        } else {
            if (max_Q_idxs.size() == 1) {
                next_act = max_Q_idxs[0];
            } else {
                next_act = max_Q_idxs[SampleDistribution::uniform_int(max_Q_idxs.size())];
            }
        }
        return next_act;
    }

    bool should_explore() {
        double rand_val = SampleDistribution::uniform_real();
        return rand_val < epsilon ? true : false;
    }

    vector<double> sample_averages() {
        vector<double> Q_estimates(testbed.K, 0);
        for (int i = 0; i < testbed.K; i++) {
            if (Kth_occurence[i] > 0) {
                Q_estimates[i] = total_reward[i] / Kth_occurence[i];
            } else {
                Q_estimates[i] = 0;
            }
        }
        return Q_estimates;
    }
};


int main() {
    int n_runs = 2000;
    int timesteps = 1000;
    int print_every = 500;
    vector<double> epsilons { 0, 0.01, 0.1 };
    vector<vector<tuple<double, int>>> all_reward_sum;

    for (double epsilon : epsilons) {
        printf("EPSILON = %.2f \n", epsilon);
        vector<tuple<double, int>> reward_at_t_sum(timesteps, make_tuple(0, 0));

        for (int i = 1; i <= n_runs; i++) {
            KArmedTestbed testbed = KArmedTestbed(10);
            ActionValueAgent agent = ActionValueAgent(testbed, epsilon);
            vector<tuple<double, int>> reward_at_t = agent.single_run();

            for (int t = 0; t < timesteps; t++) {
                get<0>(reward_at_t_sum[t]) += get<0>(reward_at_t[t]);
                get<1>(reward_at_t_sum[t]) += get<1>(reward_at_t[t]);
            }  
            
            if (i % print_every == 0) {
                printf("Run %d/%d \n", i, n_runs);
            }
        }
        all_reward_sum.push_back(reward_at_t_sum);
    }

    printf("Writing results to file... \n");
    ofstream csv_file("10arm_average_rewards_t.csv");
    for (int t = 0; t < timesteps; t++) {
        csv_file << get<0>(all_reward_sum[0][t]) / 2000.0 << ", ";
        csv_file << get<0>(all_reward_sum[1][t]) / 2000.0 << ", ";
        csv_file << get<0>(all_reward_sum[2][t]) / 2000.0 << ", ";
        csv_file << get<1>(all_reward_sum[0][t]) / 2000.0 << ", ";
        csv_file << get<1>(all_reward_sum[1][t]) / 2000.0 << ", ";
        csv_file << get<1>(all_reward_sum[2][t]) / 2000.0 << endl;
    }
    csv_file.close();
}