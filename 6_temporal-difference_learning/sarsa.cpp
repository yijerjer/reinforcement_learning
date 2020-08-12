#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
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

    static int uniform_int(int min, int max) {
        uniform_int_distribution<int> uni_dist(min, max);
        return uni_dist(generator);
    }
};
random_device SampleDistribution::rd;
mt19937 SampleDistribution::generator(SampleDistribution::rd());


class GridWorld {
    tuple<int, int> grid_size;
    vector<vector<int>> wind;
    bool stochastic;
    
    public:
    tuple<int, int> start_cell;
    tuple<int, int> end_cell;
    tuple<int, int> curr_cell;
    vector<tuple<int, int>> path;

    GridWorld(bool stochastic_val = false) {
        stochastic = stochastic_val;
        int rows = 7, cols = 10;
        grid_size = make_tuple(rows, cols);
        set_wind(rows, cols);
        start_cell = make_tuple(3, 0);
        end_cell = make_tuple(3, 7);
    }

    void set_wind(int rows, int cols) {
        wind = vector<vector<int>>(rows, vector<int>(cols, 0));
        ofstream csv_file("csvs/wind_0.csv");
        
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int wind_speed = 0;
                vector<int> wind_1_cols {3, 4, 5, 8};
                vector<int> wind_2_cols {6, 7};
                if (find(wind_1_cols.begin(), wind_1_cols.end(), col) != wind_1_cols.end()) {
                    wind_speed = 1;
                } else if (find(wind_2_cols.begin(), wind_2_cols.end(), col) != wind_2_cols.end()) {
                    wind_speed = 2;
                }
                wind[row][col] = wind_speed;

                
                string end_line = (col == (cols - 1)) ? "\n" : ", ";
                csv_file << wind_speed << end_line;
            }
        }

        csv_file.close();
        
    }

    tuple<int, int> set_start_cell() {
        curr_cell = start_cell;
        path = vector<tuple<int, int>>(0);
        path.push_back(curr_cell);
        return curr_cell;
    }

    tuple<int, int> move_cell(tuple<int, int> action) {
        int vert = get<0>(action), ho = get<1>(action);
        int curr_row = get<0>(curr_cell), curr_col = get<1>(curr_cell);
        int vert_wind = get_wind(curr_row, curr_col);
        int next_row = curr_row + vert + vert_wind, next_col = curr_col + ho;

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
        path.push_back(curr_cell);
        return curr_cell;
    }

    int get_wind(int row, int col) {
        int vert_wind = wind[row][col];
        if (stochastic && vert_wind != 0) {
            double wind_proba = SampleDistribution::uniform_real(0, 1);
            if (wind_proba < 1/3) {
                vert_wind += 1;
            } else if (wind_proba > 2/3) {
                vert_wind -= 1;
            }
        }
        return vert_wind;
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


class Sarsa {
    double alpha, epsilon;
    int n_episodes;
    GridWorld grid;
    map<tuple<tuple<int, int>, tuple<int, int>>, double> Q_ests;
    
    public:
    Sarsa() {}
    Sarsa(GridWorld grid_world, int n_episodes_val, double alpha_val = 0.1, double epsilon_val = 0.1) {
        grid = grid_world;
        n_episodes = n_episodes_val;
        alpha = alpha_val;
        epsilon = epsilon_val;
    }

    void iterate() {
        for (int epi = 0; epi < n_episodes; epi++) {
            grid.set_start_cell();
            tuple<int, int> curr_state = grid.set_start_cell();
            tuple<int, int> action = policy(curr_state);
            tuple<int, int> next_state, next_action;

            int epi_length = 0;
            do {
                next_state = grid.move_cell(action);
                next_action = policy(next_state);
                double reward = -1.0;

                initialise_Q_ests(curr_state, action);
                initialise_Q_ests(next_state, next_action);
                
                tuple<tuple<int, int>, tuple<int, int>> curr_sa_pair = make_tuple(curr_state, action);
                tuple<tuple<int, int>, tuple<int, int>> next_sa_pair = make_tuple(next_state, next_action);

                Q_ests[curr_sa_pair] += alpha*(reward + Q_ests[next_sa_pair] - Q_ests[curr_sa_pair]);

                curr_state = next_state;
                action = next_action;
                epi_length++;
            } while (curr_state != grid.end_cell);

            printf("Episode %d, Length of %d, Path %d \n", epi, epi_length, (int) grid.path.size());
        }
        grid.save_path(n_episodes);
    }

    tuple<int, int> policy(tuple<int, int> state) {
        vector<tuple<int, int>> all_actions = possible_actions();
        double proba = SampleDistribution::uniform_real(0, 1);

        if (proba < epsilon) {
            int action_idx = SampleDistribution::uniform_int(0, all_actions.size() - 1);
            return all_actions[action_idx];
        } else {
            vector<tuple<int, int>> max_actions;
            double max_action_value = -pow(10, 11);

            for (auto action : all_actions) {
                initialise_Q_ests(state, action);
                tuple<tuple<int, int>, tuple<int, int>> sa_pair = make_tuple(state, action);
                if (Q_ests[sa_pair] == max_action_value) {
                    max_actions.push_back(action);
                } else if (Q_ests[sa_pair] > max_action_value) {
                    max_action_value = Q_ests[sa_pair];
                    max_actions.clear();
                    max_actions.push_back(action);
                }
            }

            int action_idx = SampleDistribution::uniform_int(0, max_actions.size() - 1);
            return max_actions[action_idx];
        }        
    }

    vector<tuple<int, int>> possible_actions() {
        vector<tuple<int, int>> actions {
            make_tuple(-1, -1), 
            make_tuple(-1, 0), 
            make_tuple(-1, 1), 
            make_tuple(0, -1), 
            make_tuple(0, 0), 
            make_tuple(0, 1), 
            make_tuple(1, -1), 
            make_tuple(1, 0), 
            make_tuple(1, 1) 
        };
        return actions;
    }

    void initialise_Q_ests(tuple<int, int> state, tuple<int, int> action) {
        if (Q_ests.find(make_tuple(state, action)) == Q_ests.end()) {
            Q_ests[make_tuple(state, action)] = (state == grid.end_cell) ? 0 : -pow(10, 10);
        }
    }
};


int main() {
    int n_episodes = 1000;
    GridWorld grid = GridWorld(true);
    Sarsa sarsa = Sarsa(grid, n_episodes);
    sarsa.iterate();
    // grid.save_path(n_episodes);
}



