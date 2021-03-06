#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <tuple>
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
};
random_device SampleDistribution::rd;
mt19937 SampleDistribution::generator(SampleDistribution::rd());


class RaceTrack {
    public:
    vector<vector<int>> track;
    vector<tuple<int, int>> start_line, finish_line;
    tuple<int, int, int, int> curr_state;
    double chaos;

    RaceTrack(double chaos_val = 0.1) {
        chaos = chaos_val;
        set_big_track();
        
    }


    void set_big_track() {
        track = vector<vector<int>>(34, vector<int>(19, 0));

        // track
        vector<tuple<int, int>> track_start_length{
            tuple<int, int>(4, 6), tuple<int, int>(4, 6),
            tuple<int, int>(3, 7), tuple<int, int>(3, 7), tuple<int, int>(3, 7), tuple<int, int>(3, 7), tuple<int, int>(3, 7), tuple<int, int>(3, 7), tuple<int, int>(3, 7),
            tuple<int, int>(2, 8), tuple<int, int>(2, 8), tuple<int, int>(2, 8), tuple<int, int>(2, 8), tuple<int, int>(2, 8), tuple<int, int>(2, 8), tuple<int, int>(2, 8), tuple<int, int>(2, 8),
            tuple<int, int>(1, 9), tuple<int, int>(1, 9), tuple<int, int>(1, 9), tuple<int, int>(1, 9), tuple<int, int>(1, 9), tuple<int, int>(1, 9), tuple<int, int>(1, 9),
            tuple<int, int>(1, 10), tuple<int, int>(1, 16),  tuple<int, int>(1, 16),  tuple<int, int>(2, 15),  tuple<int, int>(3, 14),  tuple<int, int>(3, 14),  tuple<int, int>(4, 13)
        };
        int track_row = 2;
        for (auto tup : track_start_length) {
            int start = get<0>(tup); int length = get<1>(tup);
            for (int i = start; i < start + length; i++) {
                track[track_row][i] = 1;
            }
            track_row++;
        }
        
        // finish line
        for (int i = 27; i < 33; i++) {
            track[i][17] = 3;
            finish_line.push_back(tuple<int, int>(i, 17));
        }
        // start line
        for (int i = 4; i < 10; i++) {
            track[1][i] = 2;
            start_line.push_back(tuple<int, int>(1, i));
        }
    }

    void set_small_track() {
        track = vector<vector<int>>(5, vector<int>(7, 0));
        vector<tuple<int, int>> track_start_length{
            tuple<int, int>(2, 3), tuple<int, int>(2, 3), tuple<int, int>(2, 3)
        };
        int track_row = 1;
        for (auto tup : track_start_length) {
            int start = get<0>(tup); int length = get<1>(tup);
            for (int i = start; i < start + length; i++) {
                track[track_row][i] = 1;
            }
            track_row++;
        }

        for (int i = 1; i < 4; i++) {
            track[i][5] = 3;
            finish_line.push_back(tuple<int, int>(i, 5));
        } 

        for (int i = 1; i < 4; i++) {
            track[i][1] = 2;
            start_line.push_back(tuple<int, int>(i, 1));
        }
    }

    tuple<int, int, int, int> set_start_state() {
        int start_idx = SampleDistribution::uniform_int(0, start_line.size() - 1);
        tuple<int, int> start_loc = start_line[start_idx];
        curr_state = make_tuple(get<0>(start_loc), get<1>(start_loc), 0, 0);
        return curr_state;
    }

    tuple<int, int, int, int> move_state(tuple<int, int> action) {
        double chaos_proba = SampleDistribution::uniform_real();
        
        if (chaos_proba < chaos) {
            return curr_state;
        } else {
            tuple<int, int> new_vel = make_tuple(get<2>(curr_state) + get<0>(action), get<3>(curr_state) + get<1>(action));
            vector<tuple<int, int>> path = get_path(new_vel);

            for (auto cell : path) {
                int cell_type = track[get<0>(cell)][get<1>(cell)];
                if (cell_type == 0) {
                    set_start_state();
                    return curr_state;
                } else if (cell_type == 3) {
                    curr_state = make_tuple(get<0>(cell), get<1>(cell), get<0>(new_vel), get<1>(new_vel));
                    return curr_state;
                }
            }

            get<0>(curr_state) += get<0>(new_vel);
            get<1>(curr_state) += get<1>(new_vel);
            get<2>(curr_state) = get<0>(new_vel);
            get<3>(curr_state) = get<1>(new_vel);
            return curr_state;
        }

    }

    vector<tuple<int, int>> get_path(tuple<int, int> vel) {
        double vert_v = get<0>(vel); double ho_v = get<1>(vel);
        double larger_v = (vert_v > ho_v) ? vert_v : ho_v;
        vector<tuple<int, int>> path;

        for (int i = 1; i <= larger_v; i++) {
            double vert_trajec = i * (vert_v / larger_v);
            double ho_trajec = i * (ho_v / larger_v);

            int vert_path = get<0>(curr_state) + round(vert_trajec);
            int ho_path = get<1>(curr_state) + round(ho_trajec);
            path.push_back(tuple<int, int>(vert_path, ho_path));
        }

        return path;
    }

    void show_track(vector<tuple<int, int, int, int>> states = vector<tuple<int, int, int, int>>(0), string csv_filename = "") {
        ofstream csv_file;
        if (csv_filename != "") {
            csv_file = ofstream(csv_filename);
        }
        
        vector<tuple<int, int>> path;
        for (auto state : states) {
            path.push_back(tuple<int, int>(get<0>(state), get<1>(state)));
        }
        
        for (int v = 0; v < track.size(); v++) {
            for (int h = 0; h < track[0].size(); h++) {
                if (find(path.begin(), path.end(), tuple<int, int>(v, h)) != path.end()) {
                    cout << "x ";
                    if (csv_filename != "") {
                        csv_file << "x,";
                    }
                } else {
                    if (track[v][h] == 0) {
                        cout << "- ";
                        if (csv_filename != "") {
                            csv_file << "-,";
                        }
                    } else if (track[v][h] == 1) {
                        cout << "  ";
                        if (csv_filename != "") {
                            csv_file << " ,";
                        }
                    } else {
                        cout << track[v][h] << " ";
                        if (csv_filename != "") {
                            csv_file << track[v][h] << ",";
                        }
                    }
                }
            }
            cout << endl;
            if (csv_filename != "") {
                csv_file << endl;
            }
        }
        csv_file.close();
    }

};


class OffPolicyMC {
    RaceTrack rt; 
    map<tuple<tuple<int, int, int, int>, tuple<int, int>>, double> Q_ests;
    map<tuple<tuple<int, int, int, int>, tuple<int, int>>, double> C_vals;
    map<tuple<int, int, int, int>, tuple<int, int>> target_policy;
    double gamma;
    double b_epsilon;
    int n_episodes;
    
    public:
    OffPolicyMC() {}
    OffPolicyMC(RaceTrack racetrack, int n_episodes_val = 1000, double gamma_val = 0.9, double b_epsilon_val = 0.1) {
        rt = racetrack;
        gamma = gamma_val;
        b_epsilon = b_epsilon_val;
        n_episodes = n_episodes_val;
    }

    void iterate() {
        for (int epi = 0; epi < n_episodes; epi++) {
            double G = 0.0;
            double W = 1.0;

            tuple<vector<tuple<int, int, int, int>>, vector<tuple<int, int>>> state_action_seq = generate_seq((string) "behaviour");
            vector<tuple<int, int, int, int>> state_seq = get<0>(state_action_seq);
            vector<tuple<int, int>> action_seq = get<1>(state_action_seq);

            for (int i = (state_seq.size() - 1); i >= 0; i--) {
                G = gamma * G - 1;
                tuple<tuple<int, int, int, int>, tuple<int, int>> state_action_pair = make_tuple(state_seq[i], action_seq[i]);

                C_vals[state_action_pair] += W;
                if (Q_ests.find(state_action_pair) == Q_ests.end()) {
                    Q_ests[state_action_pair] = -pow(10, 10);
                }

                Q_ests[state_action_pair] += (W / C_vals[state_action_pair]) * (G - Q_ests[state_action_pair]);
                target_policy[state_seq[i]] = get_max_action(state_seq[i]);
                if (action_seq[i] != target_policy[state_seq[i]]) {
                    break;
                }

                W = W / (1 / (double) (1 - b_epsilon + b_epsilon/9));

            }
            printf("Completed iteration %d \n", epi);
            if ((epi + 1) == 10 || (epi + 1) == 100 || (epi + 1) == 1000 || (epi + 1) == 10000) {
                vector<tuple<int, int, int, int>> path = get<0>(generate_seq("target"));
                string csv_filename = "csvs/noise_path_episode_" + to_string(epi + 1) + ".csv";
                rt.show_track(path, csv_filename);
            }
        }
    }

    tuple<vector<tuple<int, int, int, int>>, vector<tuple<int, int>>> generate_seq(string policy_type) {
        vector<tuple<int, int, int, int>> state_seq;
        vector<tuple<int, int>> action_seq;
        tuple<int, int, int, int> curr_state = rt.set_start_state();
        do {
            tuple<int, int> action;
            if (policy_type == "behaviour") {
                action = behaviour_policy(curr_state);
            } else if (policy_type == "target") {
                action = target_policy_func(curr_state);
            }
            state_seq.push_back(curr_state);
            action_seq.push_back(action);

            curr_state = rt.move_state(action);
        } while (rt.track[get<0>(curr_state)][get<1>(curr_state)] != 3);

        return make_tuple(state_seq, action_seq);
    }

    tuple<int, int> behaviour_policy(tuple<int, int, int, int> state) {
        tuple<int, int> next_action;
        double proba = SampleDistribution::uniform_real(0, 1);

        if (proba < b_epsilon) {
            vector<tuple<int, int>> possible_actions = next_possible_actions(state);
            int action_idx = SampleDistribution::uniform_int(0, possible_actions.size() - 1);
            next_action = possible_actions[action_idx];
        } else {
            next_action = get_max_action(state);
        }
        
        return next_action;
    }

    tuple<int, int> target_policy_func(tuple<int, int, int, int> state) {
        return get_max_action(state);
    }

    vector<tuple<int, int>> next_possible_actions(tuple<int, int, int, int> state) {
        vector<tuple<int, int>> possible_actions;
        vector<int> vert_actions {-1, 0, 1};
        vector<int> ho_actions {-1, 0, 1};
        int vert_v = get<2>(state); int ho_v = get<3>(state);

        if (vert_v >= 5) {
            vert_actions.erase(vert_actions.begin() + 2);
        } else if (vert_v <= 0) {
            vert_actions.erase(vert_actions.begin());
        }

        if (ho_v >= 5) {
            ho_actions.erase(ho_actions.begin() + 2);
        } else if (ho_v <= 0) {
            ho_actions.erase(ho_actions.begin());
        }

        for (int vert_ac : vert_actions) {
            for (int ho_ac : ho_actions) {
                tuple<int, int> next_vel(vert_v + vert_ac, ho_v + ho_ac);
                if (next_vel != tuple<int, int>(0, 0)) {
                    tuple<int, int> action(vert_ac, ho_ac);
                    possible_actions.push_back(action);
                }
            }
        }

        return possible_actions;
    }

    tuple<int, int> get_max_action(tuple<int, int, int, int> state) {
        vector<tuple<int, int>> possible_actions = next_possible_actions(state);
        vector<tuple<int, int>> max_actions;
        double max_action_value = -pow(10, 10);
        
        for (auto action : possible_actions) {
            tuple<tuple<int, int, int, int>, tuple<int, int>> state_action_pair = make_tuple(state, action);
            if (Q_ests.find(state_action_pair) == Q_ests.end()) {
                Q_ests[state_action_pair] = -pow(10, 10);
            }

            if (Q_ests[state_action_pair] == max_action_value) {
                max_actions.push_back(action);
            } else if (Q_ests[state_action_pair] > max_action_value) {
                max_action_value = Q_ests[state_action_pair];
                max_actions.clear();
                max_actions.push_back(action);
            }
        }

        int idx = SampleDistribution::uniform_int(0, max_actions.size() - 1);
        return max_actions[idx];
    }
};



int main() {
    RaceTrack track = RaceTrack();
    track.show_track();
    OffPolicyMC method = OffPolicyMC(track, 10000);
    method.iterate();
}