//
// Created by home on 06.06.2022.
//

#ifndef EXAMPLE_MNIST_NET_H
#define EXAMPLE_MNIST_NET_H

#include <Eigen/Dense>
#include "tensor.hpp"

struct MnistNet {

    static constexpr double epsilon = 0.12;

    ng::CPUTensor* l1;
    ng::CPUTensor* l2;

    inline MnistNet() {
        /*
            lower, upper = -(1.0 / sqrt(n)), (1.0 / sqrt(n))
            numbers = rand(1000)
            scaled = lower + numbers * (upper - lower)
         */

        auto& w1 = Eigen::MatrixXd::Random(784, 25);
        l1 = new ng::CPUTensor{w1, true};

        auto& w2 = Eigen::MatrixXd::Random(25, 10);
        l2 = new ng::CPUTensor{w2, true};
    };

    ng::CPUTensor& forward(ng::CPUTensor& x) {
        x = x * const_cast<const ng::CPUTensor&>(*l1);
        x = x.relu();
        x = x * const_cast<const ng::CPUTensor&>(*l2);
        x = x.sigmoid();

        return x;
    }
};

#endif //EXAMPLE_MNIST_NET_H
