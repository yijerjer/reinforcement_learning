#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
using namespace std;


class ValueIteration {
    int max_capital;
    vector<double> state_values;
    double heads_proba;
    double gamma;
    double theta;

    public:
    ValueIteration() {}
    ValueIteration(double p_h, int max_capital_val = 100, double theta_val = 0.1) {
        max_capital = max_capital_val;
        heads_proba = p_h;
        theta = theta_val;
        state_values = vector<double>(max_capital + 1, 0);
        state_values[0] = 0; state_values[max_capital] = 0;
    }
    ~ValueIteration() {}

    void iterate() {
        char buffer [50];
        string values_filename = "csvs/val_iter_state_values_p" + to_string(int(heads_proba * 100)) + ".csv";
        ofstream values_csv(values_filename);
        int count = 0;
        double delta;
        do {
            delta = 0;
            count++;
            values_csv << state_values[0] << ", ";
            for (int state = 1; state < max_capital; state++) {
                tuple<int, double> max_action_value = max_state_value(state);
                double state_value = get<1>(max_action_value);
                delta = max(delta, abs(state_values[state] - state_value));
                state_values[state] = state_value;

                values_csv << state_value << ", ";
            }
            values_csv << state_values[max_capital] << endl;
            printf("Iteration %d, Delta: %8.5f\n", count, delta);
        } while (delta > theta);

        values_csv.close();
    }

    vector<int> get_policy() {
        char buffer [50];
        string filename = "csvs/val_iter_policy_p" + to_string(int(heads_proba * 100)) + ".csv";
        ofstream csv_file(filename);
        vector<int> in_policy(max_capital - 1, 0);
        for (int state = 1; state < max_capital; state++) {
            tuple<int, double> max_action_value = max_state_value(state);
            in_policy[state - 1] = get<0>(max_action_value);
            csv_file << in_policy[state - 1] << endl;
        }

        csv_file.close();
        return in_policy;
    }

    tuple<int, double> max_state_value(int state) {
        int max_possible_action = min(state, max_capital - state);

        int max_action = 0;
        double max_state_value = 0;
        for (int action = 1; action <= max_possible_action; action++) {
            double state_value = get_state_value(state, action);
            if (state_value > max_state_value) {
                max_action = action;
                max_state_value = state_value;
            }
        }
        return make_tuple(max_action, max_state_value);
    }

    double get_state_value(int state, int action) {
        double state_value = 0.0;
        int next_win_state = state + action;
        int next_lose_state = state - action;
        int win_reward = (next_win_state == max_capital) ? 1 : 0;
        int lose_reward = 0;

        state_value += heads_proba * (win_reward + state_values[next_win_state]);
        state_value += (1 - heads_proba) * (lose_reward + state_values[next_lose_state]);

        return state_value;
    }
};


int main() {
    ValueIteration val_iter = ValueIteration(0.4, 100, 10e-20);
    val_iter.iterate();
    val_iter.get_policy();
}