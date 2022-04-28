
#include "tensor.hpp"

void Tensor::backward(Tensor *grad = nulltpr) {
    if (grad == nullptr) {
        grad = Tensor(
                Eigen::MatrixXd{data.rows(), data.cols()}.Ones()
        );
    }

    this->grad->data = this->grad->data + grad->data;

    for (const auto &dep : depends_on) {
        const auto &backward = dep.backward(grad->data);
        dep.tensor.backward(Tensor{backward});
    }
}