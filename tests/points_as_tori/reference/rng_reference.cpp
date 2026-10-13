// Reference draws of std::mt19937 with the libstdc++ distributions, std::shuffle and std::discrete_distribution,
// compared with points_as_tori.cpp_random by tests/points_as_tori/test_cpp_random.py (compiled with the system g++).
#include <random>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <numeric>
int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u01(0.0, 1.0), uab(-2.0, 3.0);
    std::normal_distribution<double> n1(0.0, 1.0), n2(1.0, 0.5);
    printf("U");  for (int i = 0; i < 5; i++) printf(" %.17g", u01(rng)); printf("\n");
    printf("A");  for (int i = 0; i < 7; i++) printf(" %.17g", uab(rng)); printf("\n");
    printf("N");  for (int i = 0; i < 9; i++) printf(" %.17g", n1(rng)); printf("\n");
    printf("M");  for (int i = 0; i < 4; i++) { printf(" %.17g", n2(rng)); printf(" %.17g", n1(rng)); } printf("\n");
    std::uniform_int_distribution<int> i9(0, 9), i1(0, 1), ibig(0, 100000);
    printf("I"); for (int i = 0; i < 10; i++) printf(" %d", i9(rng)); for (int i = 0; i < 5; i++) printf(" %d", i1(rng));
    for (int i = 0; i < 5; i++) printf(" %d", ibig(rng)); printf("\n");
    for (int n : {10, 7, 1, 2, 70000}) {
        std::vector<int> v(n); std::iota(v.begin(), v.end(), 0); std::shuffle(v.begin(), v.end(), rng);
        long long s = 0; for (int i = 0; i < n; i++) s += (long long)v[i] * (i + 1);
        printf("S %d %lld", n, s); for (int i = 0; i < std::min(n, 10); i++) printf(" %d", v[i]); printf("\n");
    }
    std::vector<double> w = {0.5, 2.0, 0.0, 1.25, 3.0, 1e-3};
    std::discrete_distribution<> dd(w.begin(), w.end());
    printf("D"); for (int i = 0; i < 20; i++) printf(" %d", dd(rng)); printf("\n");
    printf("U"); for (int i = 0; i < 3; i++) printf(" %.17g", u01(rng)); printf("\n");
    return 0;
}
