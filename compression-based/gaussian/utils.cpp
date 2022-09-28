
///////////////////////////////////////////////////////////////////////////////
// Some convenience functions
///////////////////////////////////////////////////////////////////////////////

constexpr double EPSILON = 1E-7;

// comparing two doubles with precision EPS
inline bool feq(double a, double b, double eps = EPSILON) { return abs(a - b) < eps; }

// converting containers to strings (for nicer output)
template <typename C, typename T = typename C::value_type>
string to_string(const C &_xs, const string &separator = ",") {
    constexpr bool is_set =
        is_same<C, set<T>>::value || is_same<C, std::unordered_set<T>>::value; // requires curly brackets

    vector<T> xs;
    xs.insert(end(xs), begin(_xs), end(_xs));
    if (is_set) {
        sort(xs.begin(), xs.end());
    }
    ostringstream result;
    result << (is_set ? "{" : "[");
    if (xs.size() > 0) {
        for (int i = 0; i < xs.size() - 1; i++)
            result << xs[i] << separator;
        result << xs[xs.size() - 1];
    }
    result << (is_set ? "}" : "]");
    return result.str();
}

// printing a vector
template <typename T> ostream &operator<<(ostream &out, const std::vector<T> &xs) {
    out << to_string(xs);
    return out;
}

// elementwise += operation
template <typename Container> void operator+=(Container &xs, const Container &ys) {
    auto px = xs.begin();
    auto py = ys.begin();
    while (px != xs.end()) {
        *px += *py;
        px++;
        py++;
    }
}

// elementwise division by a scalar
template <typename Container> void operator/=(Container &xs, const typename Container::value_type &d) {
    for (auto &x : xs)
        x /= d;
}

// shortand for squaring
template <typename T> inline T sqr(const T &x) { return x * x; }

// standard RNG for reusage
mt19937 SU_URNG = []() {
    random_device rd;
    vector<unsigned int> truely_random_data(mt19937::state_size);
    generate(truely_random_data.begin(), truely_random_data.end(), [&rd] { return rd(); });
    seed_seq seeds(truely_random_data.begin(), truely_random_data.end());
    mt19937 r_engine;
    r_engine.seed(seeds);

    return r_engine;
}();

// random from {a, a+1, ..., b-1, b} (both inclusive)
int random_int(int a, int b) { return uniform_int_distribution{a, b}(SU_URNG); }