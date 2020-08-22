#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <cmath>
using namespace std;

typedef tuple<tuple<int, int>, tuple<int, int>> state_action;
typedef tuple<double, tuple<int, int>> reward_state;

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
};
random_device SampleDistribution::rd;
mt19937 SampleDistribution::generator(SampleDistribution::rd());


class GridWorld {
    tuple<int, int> grid_size;
    
    public:
    tuple<int, int> start_cell;
    tuple<int, int> end_cell;
    tuple<int, int> curr_cell;
    vector<tuple<int, int>> blocked_cells;
    vector<tuple<int, int>> path;

    GridWorld() {}
    GridWorld(int rows, int cols) {
        grid_size = make_tuple(rows, cols);
        start_cell = make_tuple(5, 3);
        end_cell = make_tuple(0, 8);
        create_blocked_cells("initial");
    }
    ~GridWorld() {}

    void create_blocked_cells(string type) {
        blocked_cells.clear();
        if (type == "initial") {
            for (int i = 0; i < 8; i++) {
                blocked_cells.push_back(make_tuple(3, i));
            }
        } else if (type == "change") {
            for (int i = 1; i < 9; i++) {
                blocked_cells.push_back(make_tuple(3, i));
            }
        }
        print_grid();
    }

    void print_grid() {
        for (int i = 0; i < get<0>(grid_size); i++) {
            for (int j = 0; j < get<1>(grid_size); j++) {
                tuple<int, int> cell = make_tuple(i, j);
                if (cell == start_cell) {
                    cout << "S ";
                } else if (cell == end_cell) {
                    cout << "G ";
                } else if (is_blocked(cell)) {
                    cout << "B ";
                } else {
                    cout << "- ";
                }
            }
            cout << endl;
        }
    }

    tuple<int, int> set_start_cell() {
        curr_cell = start_cell;
        path = vector<tuple<int, int>>(0);
        path.push_back(curr_cell);
        return curr_cell;
    }

    reward_state move_cell(tuple<int, int> action) {
        int vert = get<0>(action), ho = get<1>(action);
        int curr_row = get<0>(curr_cell), curr_col = get<1>(curr_cell);
        int next_row = curr_row + vert, next_col = curr_col + ho;

        if (next_row < 0) {
            next_row = 0;
        } else if (next_row > (get<0>(grid_size) - 1)) {
            next_row = get<0>(grid_size) - 1;
        }

        if (next_col < 0) {
            next_col = 0;
        } else if (next_col > (get<1>(grid_size) - 1)) {
            next_col = get<1>(grid_size) - 1;
        }
        
        curr_cell = make_tuple(next_row, next_col);
        if (is_blocked(curr_cell)) {
            curr_cell = make_tuple(curr_row, curr_col);
        }
        double reward = (curr_cell == end_cell) ? 1 : 0;

        path.push_back(curr_cell);  
        return make_tuple(reward, curr_cell);
    }

    tuple<int, int> random_cell() {
        int rows = get<0>(grid_size), cols = get<1>(grid_size);
        tuple<int, int> cell;
        do {
            int rand_row = SampleDistribution::uniform_int(0, rows - 1);
            int rand_col = SampleDistribution::uniform_int(0, cols - 1);
            cell = make_tuple(rand_row, rand_col);
        } while (is_blocked(cell));
        return cell;
    }

    bool is_blocked(tuple<int, int> cell) {
        return (find(blocked_cells.begin(), blocked_cells.end(), cell) != blocked_cells.end());
    }

    void save_path(int episode_num) {
        string filename = "csvs/9_action_stochastic_path_episode_" + to_string(episode_num) + ".csv";
        ofstream csv_file(filename);
        for (auto cell : path) {
            cout << get<0>(cell) << "," << get<1>(cell) << endl;
            csv_file << get<0>(cell) << "," << get<1>(cell) << endl;
        }
        csv_file.close();
    }
};


class DynaQ {
    GridWorld grid;
    map<state_action, double> Q_ests;
    map<state_action, reward_state> model;
    map<state_action, int> last_visited;
    vector<double> n_cumu_rewards;
    double alpha, epsilon, gamma, kappa;
    int n_timesteps;
    int n_plan;
    bool is_plus, action_plus;
    
    public:
    DynaQ(GridWorld grid_world, bool is_plus_val = false, bool action_plus_val = false, double kappa_val = 0.001, 
        int n_plan_val = 40, int n_timesteps_val = 5000, 
        double alpha_val = 0.1, double epsilon_val = 0.1, double gamma_val=0.95) {
        grid = grid_world;
        is_plus = is_plus_val, action_plus = action_plus_val;
        n_plan = n_plan_val, n_timesteps = n_timesteps_val;
        alpha = alpha_val, epsilon = epsilon_val, gamma = gamma_val, kappa = kappa_val;
        n_cumu_rewards = vector<double>(n_timesteps, 0);
    }

    vector<double> iterate() {
        int episode_n = 0;
        int epi_length = 0;
        double cumulative_reward = 0;
        
        tuple<int, int> curr_state = grid.set_start_cell();
        tuple<int, int> action, next_state;
        double reward;

        for (int step = 0; step < n_timesteps; step++) {
            if (step == 1000) {
                grid.create_blocked_cells("change");
            }

            action = policy(curr_state, step);
            reward_state reward_next_state = grid.move_cell(action);
            reward = get<0>(reward_next_state); 
            next_state = get<1>(reward_next_state);
            
            state_action curr_sa_pair = make_tuple(curr_state, action);
            last_visited[curr_sa_pair] = step;
            Q_ests[curr_sa_pair] += alpha*(reward + gamma*get_max_Q(next_state) - Q_ests[curr_sa_pair]);
            model[curr_sa_pair] = make_tuple(reward, next_state);
            model_planning(step);

            curr_state = next_state;
            cumulative_reward += reward;
            n_cumu_rewards[step] = cumulative_reward;
            epi_length++;

            if (curr_state == grid.end_cell) {
                printf("Time Step %d, Episode %d, Path Length %d \n", step, episode_n, (int) grid.path.size());
                episode_n++;
                epi_length = 0;
                curr_state = grid.set_start_cell();
            }

        }
        printf("END, Episode %d, Path Length %d \n", episode_n, (int) grid.path.size());
        return n_cumu_rewards;
    }

    tuple<int, int> policy(tuple<int, int> state, int time_step) {
        vector<tuple<int, int>> all_actions = possible_actions();
        double proba = SampleDistribution::uniform_real(0, 1);

        if (proba < epsilon) {
            int action_idx = SampleDistribution::uniform_int(0, all_actions.size() - 1);
            return all_actions[action_idx];
        } else {
            vector<tuple<int, int>> max_actions;
            double max_action_value = -100;

            for (auto action : all_actions) {
                state_action sa_pair = make_tuple(state, action);
                double Q_est = Q_ests[sa_pair];
                if (action_plus) {
                    Q_est += kappa * sqrt(time_step - last_visited[sa_pair]);
                }
                
                if (Q_est == max_action_value) {
                    max_actions.push_back(action);
                } else if (Q_est > max_action_value) {
                    max_action_value = Q_est;
                    max_actions.clear();
                    max_actions.push_back(action);
                }
            }

            int action_idx = SampleDistribution::uniform_int(0, max_actions.size() - 1);
            return max_actions[action_idx];
        }        
    }

    void model_planning(int time_step) {
        for (int n = 0; n < n_plan; n++) {
            double reward;
            tuple<int, int> next_state;
            state_action rand_sa_pair = random_sa_pair();
            if (is_plus) {
                if (model.find(rand_sa_pair) == model.end()) {
                    model[rand_sa_pair] = make_tuple(0, get<0>(rand_sa_pair));
                    reward_state rs_pair = model[rand_sa_pair];
                    reward = 0;
                    next_state = get<1>(rs_pair);
                } else {
                    reward_state rs_pair = model[rand_sa_pair];
                    double tau = time_step - last_visited[rand_sa_pair];
                    reward = get<0>(rs_pair) + kappa * sqrt(tau);
                    next_state = get<1>(rs_pair);
                }
            } else {
                if (action_plus && model.find(rand_sa_pair) == model.end()) {
                    model[rand_sa_pair] = make_tuple(0, get<0>(rand_sa_pair));
                }
                reward_state rs_pair = model[rand_sa_pair];
                reward = get<0>(rs_pair);
                next_state = get<1>(rs_pair);
            }

            Q_ests[rand_sa_pair] += alpha * (reward + gamma*get_max_Q(next_state) - Q_ests[rand_sa_pair]);
        }
    }

    state_action random_sa_pair() {
        state_action rand_sa_pair;
        if (is_plus || action_plus) {
            tuple<int, int> rand_state = grid.random_cell();
            vector<tuple<int, int>> all_actions = possible_actions();
            int action_idx = SampleDistribution::uniform_int(0, all_actions.size() - 1);
            tuple<int, int> rand_action = all_actions[action_idx];
            rand_sa_pair = make_tuple(rand_state, rand_action);
        } else {
            auto iter = model.begin();
            int rand_idx = SampleDistribution::uniform_int(0, model.size() - 1);
            advance(iter, rand_idx);
            rand_sa_pair = iter->first;
        }
        return rand_sa_pair;
    }
    

    double get_max_Q(tuple<int, int> state) {
        double max_action_value = -100;
        for (auto action : possible_actions()) {
            state_action sa_pair = make_tuple(state, action);
            double Q_est = Q_ests[sa_pair];
            if (Q_ests[sa_pair] > max_action_value) {
                max_action_value = Q_ests[sa_pair];
            }
        }
        return max_action_value;
    }

    vector<tuple<int, int>> possible_actions() {
        vector<tuple<int, int>> actions {
            make_tuple(-1, 0), 
            make_tuple(0, -1), 
            make_tuple(0, 1), 
            make_tuple(1, 0), 
        };
        return actions;
    }
};


int main() {
    int n_reps = 30;
    ofstream csv_file("csvs/action_plus_rewards.csv");

    for (int n = 0; n < n_reps; n++) {
        GridWorld grid = GridWorld(6, 9);
        DynaQ dyna = DynaQ(grid, false, false);
        vector<double> rewards = dyna.iterate();
        for (int i = 0; i < rewards.size(); i++) {
            string end_line = (i == rewards.size() - 1) ? "\n" : ", ";
            csv_file << rewards[i] << end_line;
        }
    }
    csv_file.close();
}