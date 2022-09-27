#include <bits/stdc++.h>
using namespace std;

#include "utils.cpp"

using Real = long double;

#define USE_OMP // comment out to use without OpenMP

#ifdef USE_OMP
#include "omp.h"
#endif

// quantisation according to LBG algorithm
vector<vector<Real>> LBGalg(const int num_clusters, const vector<vector<Real>> &z, const bool verbose = false);

// calculating average distortion
Real ave_distortion(const int beta, const int N, const vector<vector<Real>> &z, const vector<vector<Real>> &q,
                    const bool verbose = false);

int main(int argc, char **argv) {

    if (argc != 5 && argc != 6) {
        cerr << "Called with wrong set of parameters. Format:\n";
        cerr << "  gauss_dataset_quant beta N fname_train fname_test [num_repeat]\n";
        cerr << "    where\n";
        cerr << "      beta is the number of symbols in one file\n";
        cerr << "      N is the number of files in request (i.e., leakage L=1/N)\n";
        cerr << "      fname_train filename with the training dataset\n";
        cerr << "      fname_test  filename with the test dataset\n";
        cerr << "      num_repeat  number of repetitions (default: 1)\n";

        throw "Wrong parameters";
    }

    int beta = stoi(argv[1]);
    int N = stoi(argv[2]);
    string fname_train(argv[3]);
    string fname_test(argv[4]);
    int num_repeat = (argc == 6 ? stoi(argv[5]) : 1);

    cout << "Summary of the parameters read:\n";
    cout << "  beta        " << beta << endl;
    cout << "  N           " << N << endl;
    cout << "  fname_train " << fname_train << endl;
    cout << "  fname_test  " << fname_test << endl;
    cout << "  num_repeat  " << num_repeat << endl;

    vector<vector<Real>> z_train;
    vector<vector<Real>> z_test;

    ifstream ftrain(fname_train);
    for (string line; getline(ftrain, line);) {
        istringstream iss(line);
        vector<Real> zi(N * beta, -1);
        bool read_succesfully;
        for (int i = 0; i < N * beta; i++) {
            iss >> zi[i];
        }
        z_train.push_back(std::move(zi));
    }
    ftrain.close();
    cout << "[INFO] read " << z_train.size() << " training entries" << endl;
    cout << "[INFO] penulitmate entry: " << z_train[z_train.size() - 2] << endl;
    cout << "[INFO] last entry:        " << z_train[z_train.size() - 1] << endl;

    ifstream ftest(fname_test);
    for (string line; getline(ftest, line);) {
        istringstream iss(line);
        vector<Real> zi(N * beta, -1);
        bool read_succesfully;
        for (int i = 0; i < N * beta; i++) {
            iss >> zi[i];
        }
        z_test.push_back(std::move(zi));
    }
    ftest.close();
    cout << "[INFO] read " << z_test.size() << " test entries" << endl;
    cout << "[INFO] penulimate entry: " << z_test[z_test.size() - 2] << endl;
    cout << "[INFO] last entry:       " << z_test[z_test.size() - 1] << endl;

#ifdef USE_OMP
    int num_cores = omp_get_num_procs();
#else
    int num_cores = 1;
#endif
    cout << "[INFO] there are " << num_cores << " cores available." << endl;

    vector<vector<Real>> RDcurve;
    // for (int r = 0; r <= 6; r++) {
    for (int r : {0, 2 * beta, 4 * beta}) {
        int num_clusters = (1 << r);
        cout << "=================================================" << endl;
        cout << "[INFO] r=" << r << ", num_clusters=" << num_clusters << endl;

        Real best_distortion = numeric_limits<Real>::max();

        for (int repeats = 0; repeats < num_repeat; repeats++) {
            cout << "  REPETITION " << (repeats + 1) << " OUT OF " << num_repeat << endl;
            auto q = LBGalg(num_clusters, z_train, true);
            // if (r <= 2)
            //     cout << "q: " << q << endl;
            auto test_distortion = ave_distortion(beta, N, z_test, q, true);
            best_distortion = min(best_distortion, test_distortion);
        }

        cout << "Test distortion: " << best_distortion << endl;
        RDcurve.push_back({(Real)r / beta, best_distortion});
    }

    cout << "\nRD curve found: " << RDcurve << endl;

    return 0;
}

template <typename T> T MSE(const vector<T> &a, const vector<T> &b) {
    if (a.size() != b.size()) {
        throw domain_error("sqr_distance(): sizes of vectors are not equal, " + to_string(a.size()) +
                           " != " + to_string(b.size()));
    }

    T d = 0.0;
    for (int i = 0; i < a.size(); i++)
        d += sqr(a[i] - b[i]);

    return d / a.size();
}

vector<vector<Real>> LBGalg(const int num_clusters, const vector<vector<Real>> &z, const bool verbose) {
    int n = z.size();
    if (n == 0) {
        cerr << "LBGalg: empty dataset provided" << endl;
        throw 42;
    }

    if (num_clusters < 1 || num_clusters >= n) {
        cerr << "LBGalg: num_clusters=" << num_clusters << " is incorrect" << endl;
        throw 42;
    }

    int dim = z[0].size();
    Real a = numeric_limits<Real>::max(), b = numeric_limits<Real>::min();
    for (int i = 0; i < n; i++) {
        if (z[i].size() != dim) {
            cerr << "LBGalg: dimension of z[" << i << "] is " << z[i].size() << ", while dimension of z[0] is " << dim
                 << endl;
            throw 42;
        }
        for (int j = 0; j < dim; j++) {
            a = min(a, z[i][j]);
            b = max(b, z[i][j]);
        }
    }

    vector<vector<Real>> q(num_clusters, vector<Real>(dim));
    // q is initialised with random subset of original points
    vector<int> indices(z.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), SU_URNG);
    for (int j = 0; j < num_clusters; j++)
        q[j] = z[indices[j]];

    // iterations of LBG aglorithm
    Real prev_distortion = numeric_limits<Real>::max();
    Real curr_distortion = prev_distortion / 2;

    while (!feq(prev_distortion, curr_distortion, 0.001)) {
        prev_distortion = curr_distortion;
        vector<vector<int>> neighbours(num_clusters);
        Real ave_distortion = 0;

#pragma omp parallel for reduction(+ : ave_distortion)
        // re-calculating neighbourhoods
        for (int i = 0; i < n; i++) {
            Real dist = numeric_limits<Real>::max();
            int cluster = -1;
            for (int j = 0; j < num_clusters; j++) {
                auto new_dist = MSE(z[i], q[j]);
                if (new_dist < dist) {
                    dist = new_dist;
                    cluster = j;
                }
            }

            if (cluster < 0 || cluster >= num_clusters) {
                cerr << "LBGalg: found wrong cluster " << cluster << endl;
                throw 42;
            }
#pragma omp critical
            { neighbours[cluster].push_back(i); }
            ave_distortion += dist;
        }
        ave_distortion /= n;
        if (verbose)
            cout << "distortion: " << ave_distortion << endl;

        // changing q's to centres of masses of their neighbourhoods
        for (int j = 0; j < num_clusters; j++) {
            if (neighbours[j].size() == 0) {
                cout << "[WARN] cluster " << j << " has empty neighbourood" << endl;
                // assigning q[j] to a random point so that is will have at least one neighbour at next iteration
                q[j] = z[random_int(0, n - 1)];
            } else {
                q[j].assign(dim, 0);
                for (int i : neighbours[j])
                    q[j] += z[i];
                q[j] /= neighbours[j].size();
            }
        }
        curr_distortion = ave_distortion;
    }

    return q;
}

Real ave_distortion(const int beta, const int N, const vector<vector<Real>> &z, const vector<vector<Real>> &q,
                    const bool verbose) {
    int n = z.size();
    if (n == 0) {
        cerr << "ave_distortion: empty dataset provided" << endl;
        throw 42;
    }

    int num_clusters = q.size();
    if (num_clusters < 1 || num_clusters >= n) {
        cerr << "LBGalg: num_clusters=" << num_clusters << " is incorrect" << endl;
        throw 42;
    }

    int dim = z[0].size();
    if (dim != N * beta) {
        cerr << "ave_distortion: dim=" << dim << " is not equal " << N << "*" << beta << endl;
        throw 42;
    }
    for (int i = 0; i < n; i++) {
        if (z[i].size() != dim) {
            cerr << "LBGalg: dimension of z[" << i << "] is " << z[i].size() << ", while dimension of z[0] is " << dim
                 << endl;
            throw 42;
        }
    }

    Real ave_distortion = 0;

    // re-calculating neighbourhoods
    for (int i = 0; i < n; i++) {
        Real dist = numeric_limits<Real>::max();
        for (int j = 0; j < num_clusters; j++) {
            auto new_dist = MSE(z[i], q[j]);
            if (new_dist < dist) {
                dist = new_dist;
            }
        }

        ave_distortion += dist;
    }
    ave_distortion /= n;

    return ave_distortion;
}