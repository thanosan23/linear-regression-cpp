#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <random>
#include <cassert>

constexpr int TRAIN_LENGTH = 100;
constexpr int EPOCHS = 250000;
constexpr double LEARNING_RATE = 1e-4;

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

template<typename T>
T random(T a, T b) {
  return std::uniform_int_distribution<T>(a, b)(rng);
}

double MSE(std::vector<double> Y_train, std::vector<double> Y_pred) {
  assert(Y_train.size() == Y_pred.size());
  double acc = 0;
  for(int i = 0; i < Y_train.size(); i++) {
    acc += pow((Y_train[i] - Y_pred[i]), 2);
  }
  acc *= (1.0f/(float)Y_train.size());
  return acc;
}

std::vector<double> diff(std::vector<double> a, std::vector<double> b) {
  assert(a.size() == b.size());
  std::vector<double> ret((int) a.size());
  for(int i = 0; i < (int) a.size(); i++) {
    ret[i] = (a[i] - b[i]);
  }
  return ret;
}

std::vector<double> dot(std::vector<double> a, std::vector<double> b) {
  assert(a.size() == b.size());
  std::vector<double> ret((int) a.size());
  for(int i = 0; i < (int) a.size(); i++) {
    ret[i] = a[i] * b[i];
  }
  return ret;
}

std::vector<double> dot(std::vector<double> a, double b) {
  std::vector<double> ret((int) a.size());
  for(int i = 0; i < a.size(); i++) {
    ret[i] = a[i] * b;
  }
  return ret;
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  
  std::vector<double> X_train(TRAIN_LENGTH), Y_train(TRAIN_LENGTH);
  for(int i = 0; i < TRAIN_LENGTH; i++) {
    X_train[i] = i+1;
    Y_train[i] = (i+1) * 2 + 1;
  }
  
  std::vector<double> params(2, 0.0f);
  std::vector<double> gradients(2, 0.0f);
  
  std::function<std::vector<double>(std::vector<double>&)> pred = [&](std::vector<double>& data) {
    std::vector<double> preds((int)data.size());
    for(int i = 0; i < (int)data.size(); i++) {
      preds[i] = data[i] * params[0] + params[1];
    }
    return preds;
  };
  
  for(int epoch = 0; epoch < EPOCHS; epoch++) {
    std::vector<double> Y_pred = pred(X_train);
    double loss = MSE(Y_train, Y_pred);
    
    std::vector<double> dif = diff(Y_train, Y_pred);
    std::vector<double> temp = dot(X_train, dif);
    gradients[0] = -2 * accumulate(temp.begin(), temp.end(), 0.0f) * (1.0/double(TRAIN_LENGTH));
    gradients[1] = -2 * accumulate(dif.begin(), dif.end(), 0.0f) * (1.0/(double)TRAIN_LENGTH);

    for(int i = 0; i < (int) params.size(); i++) {
      params[i] -= LEARNING_RATE * gradients[i];
    }
    std::cout << epoch+1 << ' ' << loss << '\n';
  }

  std::vector<double> X_test(5);
  for(int i = 0; i < (int) X_test.size(); i++) X_test[i] = random(110, 100000);

  std::vector<double> Y_test((int) X_test.size());
  for(int i = 0; i < (int) X_test.size(); i++) Y_test[i] = X_test[i] * 2 + 1;
  
  std::vector<double> Y_pred = pred(X_test);
  
  std::cout << "TEST DATA:\n";
  for(double& i : X_test) std::cout << i << ' ';
  std::cout << '\n';

  std::cout << "EXPECTED:\n";
  for(double& i : Y_test) std::cout << i << ' ';
  std::cout << '\n';
  
  std::cout << "PREDICTED:\n";
  for(double& i : Y_pred) std::cout << i << ' ';
  std::cout << '\n';

  std::cout << "PARAMETERS (WEIGHT, BIAS):" << '\n';
  for(double& p : params) std::cout << p << ' ';
  std::cout << '\n';
}
