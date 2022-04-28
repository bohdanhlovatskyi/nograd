#ifndef NG_TENSOR__
#define NG_TENSOR__

#include <Eigen/Dense>

namespace ng {
    class Tensor {
    private:
        Eigen::MatrixXd data;
        Eigen::MatrixXd grad;

        bool requires_grad;
        Context ctx;

    public:
        Tensor(Eigen::MatrixXd data): data{data}, grad{nullptr}, ctx{nullptr} {};

        void backward() {
            if (ctx == nullptr) {
                return
            }

            if (self.grad == nullptr) {
                // make self grad consist of zeroes with input data shape
            }

            auto grads = ctx.backward(ctx, grad);

        }
    };

    backward();



}

#endif // NG_TENSOR__