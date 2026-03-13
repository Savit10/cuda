#include <iostream>
#include <chrono>
#include <vector>
using namespace std;

int main() {

  const int N = 1500000;
  std::vector<int> arr(N);
  for (int i = 0; i < N; i++) {
      arr[i] = i + 1;
  }
  std::vector<int>arr2(N);
  for (int i = 0; i < N; i++) {
    arr2[i] = (i + 1) * 10;
  }
  std::vector<int>arr3(N);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    arr3[i] = arr[i] + arr2[i];
    //cout << "arr3[" << i << "] = " << arr3[i] << endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "CPU time: " << us << " ms\n";

  return 0;
}
