
#include "ops.hpp"

Tensor* matmul_(Tensor* t1, Tensor* t2) {
    const auto& data = t1->data * t2->data;

    std::vector<Function> depends_on;

    if (t1->requires_grad == true) {
        auto matmul1 = [](const Eigen::MatrixXd& grad){
            return grad * t2->data.transpose();
        };

        depends_on.push_back(Function{t1, matmul1});
    }

    if (t2->requires_grad == true) {
        auto matmul2 = [](const Eigen::MatrixXd& grad){
            return t1->data.transpose() * grad;
        };

        depends_on.push_back(Function{t2, matmul2});
    }

    auto output_requires_grad = t1->requires_grad || t2->requires_grad;

    return new Tensor{data, output_requires_grad, depends_on};
}