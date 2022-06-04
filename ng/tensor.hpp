#ifndef NG_TENSOR__
#define NG_TENSOR__

#include <Eigen/Dense>

#include <type_traits>
#include <vector>

namespace ng {

    using func = std::function<Eigen::MatrixXd(Eigen::MatrixXd)>;

    // forward declarations
    class Tensor;
    class Function;

    struct Function {
        // operation that was performed on a ctx
        // instantiation of a function is a context
        Tensor* ctx;
        func op;

        Function(Tensor* ctx, func op) : ctx{ctx}, op{op} {};
    };

    class Tensor {

    public:
        std::vector<Function> depends_on;
        Eigen::MatrixXd data;
        Tensor* grad;
        bool requires_grad;

        struct Device {
            virtual Tensor matmul(Tensor& lhs, const Tensor& rhs) = 0;

            inline Device() = default;
        };


        struct CPUDevice : public Device {
            CPUDevice() = default;

            inline Tensor matmul(Tensor& lhs, const Tensor& rhs) override {
                std::vector<Function> depends_on;
                depends_on.reserve(lhs.requires_grad + rhs.requires_grad);

                auto& d = lhs.data * rhs.data;
                if (lhs.requires_grad) {
                    depends_on.emplace_back(
                            &lhs,
                            [rhs](Eigen::MatrixXd grad) {
                                return Eigen::MatrixXd{grad * rhs.data.transpose()};
                            });
                }

                if (rhs.requires_grad) {
                    depends_on.emplace_back(
                            &const_cast<decltype(lhs)>(rhs),
                            [lhs](Eigen::MatrixXd grad){
                                return Eigen::MatrixXd{lhs.data.transpose() * grad};
                            });
                }

                return Tensor{d,
                              lhs.requires_grad || rhs.requires_grad,
                              depends_on,
                              lhs.dev};
            }
        };

        Device* dev;


        inline Tensor() = default;

        //                      Device* dev = new CPU{}) : \
        // TODO: this makes copy for now
        inline Tensor(Eigen::MatrixXd data,
                      bool requires_grad = false,
                      const std::vector<Function>& depends_on = std::vector<Function>{},
                        Device* dev = new CPUDevice{}) : \
                         data{data}, depends_on{depends_on},
                         dev{dev}, requires_grad{requires_grad} {
            if (requires_grad) {
                zero_grad();
            }
        };

        inline Tensor& T() {
            data = data.transpose();
            return *this;
        }

        friend inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
            return os << t.data;
        }

        inline void zero_grad() {
            grad = new Tensor(Eigen::MatrixXd{data.rows(), data.cols()});
        }

        inline void backward(Tensor* grad = nullptr) {
            if (grad == nullptr) {
                grad = new Tensor(
                        Eigen::MatrixXd().setOnes(data.rows(), data.cols())
                );
            }

            if (this->grad == nullptr) {
                std::cerr << "Could not perform backward on a tensor that does not require gradient" << std::endl;
                exit(1);
            }

            assert(this->grad != nullptr);
            assert(this->grad->data.size() == grad->data.size());

            this->grad->data += grad->data;

            for (auto &dep : depends_on) {
                func f{dep.op};
                auto gr = grad->data;
                auto r = f(gr);
                dep.ctx->backward(new Tensor{r});
            }
        }

        template<typename Tn>
        friend inline Tensor operator*(Tn&& lhs, Tn&& rhs)
        {
            return lhs.dev->matmul(std::forward<Tn>(lhs), std::forward<Tn>(rhs));
        }

    };
};

#endif // NG_TENSOR__