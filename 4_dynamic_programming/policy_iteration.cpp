#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <chrono>
using namespace std;


class PolicyIteration {
    vector<vector<double>> state_values;
    vector<vector<double>> policy;
    int n_cars;
    double theta;
    bool modified;

    public:
    double gamma;
    PolicyIteration() {}
    PolicyIteration(int n, double gamma_val = 0.9, double theta_val = 0.5, bool modified_val = false) {
        n_cars = n;
        gamma = gamma_val;
        theta = theta_val;
        modified = modified_val;
        state_values = vector<vector<double>>(n + 1, vector<double>(n + 1, 0));
        policy = vector<vector<double>>(n + 1, vector<double>(n + 1, 0));
    }
    ~PolicyIteration() {}

    void iterate() {
        bool policy_stable = true;
        int count = 0;
        save_state_values(0);
        save_policy(0);
        do {
            count++;
            printf("ITERATION %d \n", count);
            evaluation();
            save_state_values(count);
            policy_stable = improvement();
            save_policy(count);
        } while (!policy_stable);
    }

    void load_state_values(string filename) {
        ifstream csv_file(filename);

    }


    // private:

    void evaluation() {
        double delta;
        int count = 0;
        chrono::steady_clock::time_point start = chrono::steady_clock::now();
        do {
            delta = 0;
            count++;
            for (int A = 0; A <= n_cars; A++) {
                for (int B = 0; B <= n_cars; B++) {
                    double old_value = state_values[A][B];
                    state_values[A][B] = new_state_value(make_tuple(A, B), policy[A][B]);
                    delta = max(delta, abs(state_values[A][B] - old_value));
                }
            }
            printf("- Evaluation: Round %2.d, Delta: %7.3f\n", count, delta);
        } while (delta > theta);
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        int time_diff = chrono::duration_cast<chrono::seconds>(end - start).count();
        printf("- Evaluation: Time elapsed %d min %d secs \n", int(floor(time_diff / 60)), time_diff % 60);
    }

    bool improvement() {
        bool policy_stable = true;
        printf("- Improvement: Searching for new policy...\n");
        for (int A = 0; A <= n_cars; A++) {
            for (int B = 0; B <= n_cars; B++){
                int max_move = 0;
                double max_state_value = 0;
                for (int move = -5; move <= 5; move++) {
                    double value = new_state_value(make_tuple(A, B), move);
                    if (value > max_state_value) {
                        max_move = move;
                        max_state_value = value;
                    }
                }

                if (max_move != policy[A][B]) {
                    policy_stable = false;
                    policy[A][B] = max_move;
                }
            }
        }
        printf("- Improvement: Policy is %s \n", policy_stable ? "stable" : "unstable");
        return policy_stable;
    }

    int get_reward(tuple<int, int> rented, int moved) {
        int total_reward = 0;
        total_reward += (get<0>(rented) + get<1>(rented)) * 10;
        total_reward -= abs(moved) * 2;

        return total_reward;
    }

    int get_modified_reward(tuple<int, int> next_state, tuple<int, int> rented, int moved) {
        int total_reward = 0;
        total_reward += (get<0>(rented) + get<1>(rented)) * 10;
        total_reward -= abs((moved > 0) ? moved - 1 : moved) * 2;

        if ((get<0>(next_state) > 10)) {
            total_reward -= 4;
        }
        if ((get<1>(next_state) > 10)) {
            total_reward -= 4;
        }

        return total_reward;
    }

    double poisson(int n, int max_n, double lambda) {
        double proba = 0;
        if (n == max_n) {
            proba = 1;
            for (int i = 0; i < max_n; i++) {
                proba -= poisson(i, lambda);
            }
        } else if (n < max_n) {
            proba = poisson(n, lambda);
        }
        return proba;
    }

    double poisson(int n, double lambda) {
        double probability = pow(lambda, n) * exp(-lambda);
        for (double i = 1; i <= n; i++) {
            probability /= i;
        }
        return probability;
    }

    double new_state_value(tuple<int, int> state, int move) {
        int A = get<0>(state);
        int B = get<1>(state);
        double state_value = 0.0;
        
        A -= move;
        B += move;
        // printf("HERE: %d %d\n", A, B);
        for (int rent_A = 0; rent_A <= A; rent_A++) {
            for (int rent_B = 0; rent_B <= B; rent_B++) {
                for (int return_A = 0; return_A <= (n_cars - (A - rent_A)); return_A++) {
                    for (int return_B = 0; return_B <= (n_cars - (B - rent_B)); return_B++) {
                        double proba = poisson(rent_A, A, 3) *
                            poisson(rent_B, B, 4) *
                            poisson(return_A, n_cars - (A - rent_A), 3) *
                            poisson(return_B, n_cars - (B - rent_B), 2);
                        int next_A = A - rent_A + return_A;
                        int next_B = B - rent_B + return_B;

                        double reward;
                        if (modified) {
                            reward = get_modified_reward(make_tuple(next_A, next_B), make_tuple(rent_A, rent_B), move);
                        } else {
                            reward = get_reward(make_tuple(rent_A, rent_B), move);
                        }
                        state_value += proba * (reward + gamma * state_values[next_A][next_B]);
                        // printf("State: (%d, %d), Next: (%d, %d), Rent: (%d, %d), Return: (%d, %d), %.5f %.5f %.5f \n", 
                        //     get<0>(state), get<1>(state), A-rent_A+return_A, B-rent_B+return_B, rent_A, rent_B, return_A, return_B, proba, reward, state_value);
                        // printf("%d, %d, %d, %d \n", rent_A, rent_B, return_A, return_B);
                        
                    }
                }
            }
        }
        return state_value;
    }

    void save_state_values(int iter) {
        string mod_string = (modified ? "mod_" : "");
        string file_name = "csvs/state_values_" + mod_string + to_string(iter) + ".csv";
        cout << "- Saving state values in " << file_name << endl; 
        ofstream csv_file(file_name);
        for (int A = 0; A <= n_cars; A++) {
            for (int B = 0; B <= n_cars; B++) {
                string end_line = (B == n_cars) ? "\n" : ", ";
                csv_file << state_values[A][B] << end_line;
            }
        }
        csv_file.close();
    }

    void save_policy(int iter) {
        string mod_string = (modified ? "mod_" : "");
        string file_name = "csvs/policy_" + mod_string + to_string(iter) + ".csv";
        cout << "- Saving policy in " << file_name << endl; 
        ofstream csv_file(file_name);
        for (int A = 0; A <= n_cars; A++) {
            for (int B = 0; B <= n_cars; B++) {
                string end_line = (B == n_cars) ? "\n" : ", ";
                csv_file << policy[A][B] << end_line;
            }
        }
        csv_file.close();
    }
};


int main() {
    PolicyIteration policy_iter = PolicyIteration(20, 0.9, 0.5, true);
    policy_iter.iterate();


}