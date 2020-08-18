#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <cmath>
using namespace std;


class SampleDistribution {
    static random_device rd;
    static mt19937 generator;

    public:
    static double uniform_real(double min = 0, double max = 1) {
        uniform_real_distribution<double> uni_dist(min, max);
        return uni_dist(generator);
    }

    static int uniform_int(int min, int max) {
        uniform_int_distribution<int> uni_dist(min, max);
        return uni_dist(generator);
    }

    static double normal(double mean = 0, double variance = 1) {
        normal_distribution<double> norm_dist(mean, variance);
        return norm_dist(generator);
    }
};
random_device SampleDistribution::rd;
mt19937 SampleDistribution::generator(SampleDistribution::rd());


typedef map<tuple<int, int>, tuple<vector<int>, vector<double>>> sa_to_sr;
class Environment {
    
    public:
    sa_to_sr sa_sr; 
    int n_states, b_factor;
    int curr_state, end_state;
    
    Environment() {}
    Environment(int n_states_val, int b_factor_val) {
        n_states = n_states_val;
        b_factor = b_factor_val;
        end_state = n_states;
        set_sa_transitions();
    }
    Environment(sa_to_sr sa_sr_copy, int n_states_val, int b_factor_val) {
        n_states = n_states_val;
        b_factor = b_factor_val;
        end_state = n_states;
        sa_sr = sa_sr_copy;
    }

    void set_sa_transitions() {
        auto set_to_states = [=](int state, int action) { 
            vector<int> rand_states(b_factor);
            vector<double> rewards(b_factor);
            generate(rand_states.begin(), rand_states.end(), [=]() { return SampleDistribution::uniform_int(0, n_states - 1); });
            generate(rewards.begin(), rewards.end(), [=]() { return SampleDistribution::normal(0, 1); });
            sa_sr[make_tuple(state, action)] = make_tuple(rand_states, rewards);
        };
        
        for (int i = 0; i < n_states; i++) {
            set_to_states(i, 0);
            set_to_states(i, 1);
        }
    }

    tuple<vector<int>, vector<double>> get_next_srs(int state, int action) {
        return sa_sr[make_tuple(state, action)];
    }

    void set_start_state() {
        curr_state = 0;
    }

    double move_with_reward(int action) {
        double end_proba = SampleDistribution::uniform_real(0, 1);
        double reward;
        
        if (end_proba < 0.1) {
            curr_state = end_state;
            reward = 0;
            
        } else {
            tuple<vector<int>, vector<double>> state_reward = sa_sr[make_tuple(curr_state, action)];
            vector<int> next_states = get<0>(state_reward);
            vector<double> rewards = get<1>(state_reward);
            int next_state_idx = SampleDistribution::uniform_int(0, b_factor - 1);
            // printf("(%d, %d), %d, %d, %d \n", curr_state, action, next_state_idx, (int) next_states.size(), (int) rewards.size());
            curr_state = next_states[next_state_idx];
            reward = rewards[next_state_idx];
        }

        return reward;
        
    }

    bool is_end() {
        return curr_state == end_state;
    }
};



class Agent {
    Environment env, sim_env;
    map<tuple<int, int>, double> Q_ests;
    double epsilon;
    int n_steps;
    int curr_step_count;
    string plan_dist;
    
    public:
    vector<double> greedy_values;

    Agent() {}
    Agent(Environment environment, string plan_dist_val = "uniform", double epsilon_val = 0.1, int n_steps_val = 5000) {
        env = environment;
        epsilon = epsilon_val;
        plan_dist = plan_dist_val;
        n_steps = n_steps_val;
        greedy_values = vector<double>(0);
        if (plan_dist == "on-policy") {
            sim_env = Environment(env.sa_sr, env.n_states, env.b_factor);
        }
    }


    void iterate() {
        curr_step_count = 0;
        do {
            model_planning();
        } while (curr_step_count < n_steps);
    }


    int e_greedy_policy(int state) {
        int action;
        double explore_proba = SampleDistribution::uniform_real(0, 1);
        if (explore_proba < 0.1) {
            action = SampleDistribution::uniform_int(0, 1);
        } else {        
            if (Q_ests[make_tuple(state, 0)] == Q_ests[make_tuple(state, 1)]) {
                action = SampleDistribution::uniform_int(0, 1);
            } else {
                action = Q_ests[make_tuple(state, 0)] > Q_ests[make_tuple(state, 1)] ? 0 : 1;
            }
        }
        return action;
    }

    void compute_greedy_value() {
        int n_reps = 100;
        double value = 0;
        
        for (int n = 0; n < n_reps; n++) {
            env.set_start_state();
            do {
                int action;
                if (Q_ests[make_tuple(env.curr_state, 0)] == Q_ests[make_tuple(env.curr_state, 1)]) {
                    action = SampleDistribution::uniform_int(0, 1);
                } else if (Q_ests[make_tuple(env.curr_state, 0)] > Q_ests[make_tuple(env.curr_state, 1)]) {
                    action = 0;
                } else {
                    action = 1;
                }

                double reward = env.move_with_reward(action);
                value += reward;
                
            } while (!env.is_end());

        }
        value /= n_reps;
        greedy_values.push_back(value);
    }

    void model_planning() {
        if (plan_dist == "uniform") {
            for (int s = 0; s < env.n_states; s++) {
                for (int a = 0; a < 2; a++) {
                    Q_ests[make_tuple(s, a)] = expected_Q_est(s, a);
                    
                    curr_step_count++;
                    if (curr_step_count % 50 == 0) {
                        compute_greedy_value();
                    }
                }
            }
        } else if (plan_dist == "on-policy") {
            sim_env.set_start_state();
            do {
                int action = e_greedy_policy(sim_env.curr_state);
                Q_ests[make_tuple(sim_env.curr_state, action)] = expected_Q_est(sim_env.curr_state, action);
                sim_env.move_with_reward(action);
                curr_step_count++;
                if (curr_step_count % 50 == 0) {
                    compute_greedy_value();
                }
            } while (!sim_env.is_end());
        }
    }

    double expected_Q_est(int s, int a) {
        double Q_sum = 0;
        tuple<vector<int>, vector<double>> next_srs = env.get_next_srs(s, a);
        for (int i = 0; i < env.b_factor; i++) {
            int next_state = get<0>(next_srs)[i];
            double reward = get<1>(next_srs)[i];

            double max_Q;
            if (Q_ests[make_tuple(next_state, 0)] == Q_ests[make_tuple(next_state, 1)]) {
                int rand_a = SampleDistribution::uniform_int(0, 1);
                max_Q = Q_ests[make_tuple(next_state, rand_a)];
            } else {
                int action = Q_ests[make_tuple(next_state, 0)] > Q_ests[make_tuple(next_state, 1)] ? 0 : 1;
                max_Q = Q_ests[make_tuple(next_state, action)];
            }

            Q_sum += reward + max_Q;
        }
        return Q_sum / (double) env.b_factor;
    }
};

void calculate_and_save_values(string plan_dist, int b, int n_states = 1000) {
    int n_iters = 100;
    vector<double> average_values;

    cout << "Calculating values for distribution: " + plan_dist + ", b: " + to_string(b) << endl; 
    for (int n = 0; n < n_iters; n++) {
        Environment env = Environment(n_states, b);
        Agent agent = Agent(env, plan_dist);
        agent.iterate();

        if (average_values.size() == 0) {   
            average_values = agent.greedy_values;
        } else {
            for (int i = 0; i < average_values.size(); i++) {
                average_values[i] += 1.0 / (n + 1.0) * (agent.greedy_values[i] - average_values[i]);
            }
        }

        if (n % 10 == 0) {
            printf("Completed Iteration %d \n", n);
        }
    }

    string filename = "csvs/" + plan_dist + "_b" + to_string(b) + "_values_" + to_string(n_states) + ".csv";
    ofstream csv_file(filename);
    for (int i = 0; i < average_values.size(); i++) {
        string end = (i == average_values.size() - 1) ? "\n" : ", ";
        csv_file << average_values[i] << end;
    }
    csv_file.close();
}


int main() {
    calculate_and_save_values("uniform", 1);
    calculate_and_save_values("uniform", 3);
    calculate_and_save_values("uniform", 10);

    calculate_and_save_values("on-policy", 1);
    calculate_and_save_values("on-policy", 3);
    calculate_and_save_values("on-policy", 10);

    calculate_and_save_values("uniform", 3, 10000);
    calculate_and_save_values("on-policy", 3, 10000);
}