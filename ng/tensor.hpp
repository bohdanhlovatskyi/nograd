#ifndef NG_TENSOR__
#define NG_TENSOR__

#include <Eigen/Dense>

#include <vector>

namespace ng {
    class Tensor {
    private:
        Eigen::MatrixXd data;
        Tensor *grad;
        std::vector<Function> depends_on;
        bool requires_grad;

    public:
        inline Tensor(Eigen::MatrixXd data, bool requires_grad = false) : \
                         data{data}, grad{nullptr}, ctx{nullptr},
                         requires_grad{requires_grad} {
            if (requires_grad) {
                zero_grad();
            }
        };

        inline void zero_grad() {
            grad = Tensor(Eigen::MatrixXd{data.rows(), data.cols()}.Zero());
        }

        void backward(Tensor *grad = nulltpr);
    };
}

#endif // NG_TENSOR__