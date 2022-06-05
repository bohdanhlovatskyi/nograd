// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>

#include <Eigen/Dense>

#include "ng/tensor2.hpp"

int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    Eigen::MatrixXd aa{2,1};
    aa(0,0) = 2.;
    aa(1,0) = 3.;
    std::cout << aa << std::endl;

    Eigen::MatrixXd bb{2,1};
    bb(0,0) = 6.;
    bb(1,0) = 2.;
    std::cout << bb << std::endl;

    ng::Tensor a{aa.transpose(), true};
    ng::Tensor b{bb, true};

    auto z = a*b;
    std::cout << z->data() << std::endl;

//    z.backward();

    std::cout << a.grad() << std::endl;
    std::cout << b.grad() << std::endl;

    return 0;
}
