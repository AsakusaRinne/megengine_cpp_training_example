#include "mainfile/train.h"

#include <cstring>
#include <iostream>

int main(int argc, char **argv) {

  std::cout << "Start to run training examples, " << argc - 1
            << " examples are specified." << std::endl;

  for (int i = 1; i < argc; i++) {
    auto name = argv[i];
    if (strcmp(name, "mnist") == 0) {
      train_mnist();
    } else {
      std::cout << "No example named " << name << "." << std::endl;
    }
  }

  std::cout << "Finish." << std::endl;

  return 0;
}