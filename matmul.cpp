#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    int N = 16;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    int size = N * N;

    std::vector<float> A(size), B(size), C(size);
    for (int i = 0; i < size; i++) {
        A[i] = float(i + 1);
        B[i] = float(i + 1);
    }

    auto start = high_resolution_clock::now();

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    auto end = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(end - start).count();
    cout << us << endl;  // Output just the time in microseconds

    return 0;
}
