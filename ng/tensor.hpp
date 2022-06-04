#ifndef NG_TENSOR__
#define NG_TENSOR__

#include <Eigen/Dense>

#include <type_traits>
#include <vector>

namespace ng {

    using func = std::function<void(void)>;

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
        bool requires_grad;

        struct Device {
            inline Device() = default;

            void* data;
            void* grad;

            virtual void T() = 0;
            virtual void zero_grad() = 0;
            virtual Tensor matmul(Tensor& lhs, const Tensor& rhs) = 0;
        };

        struct CPUDevice : public Device {
            Eigen::MatrixXd data;
            Eigen::MatrixXd grad;

            CPUDevice() = default;
            CPUDevice(Eigen::MatrixXd d) : data{d} {};

            inline Eigen::MatrixXd& getData() {
                return data;
            }

            inline void zero_grad() override {
                grad = Eigen::MatrixXd().setZero(data.rows(), data.cols());
            }

            inline void T() {
                auto& dlhs = reinterpret_cast<CPUDevice*>(this)->data;
                dlhs = dlhs.transpose();
            }

            Tensor matmul(Tensor& lhs, const Tensor& rhs) override {
                std::vector<Function> depends_on;
                depends_on.reserve(lhs.requires_grad + rhs.requires_grad);

                auto& dlhs = reinterpret_cast<CPUDevice*>(lhs.dev)->data;
                auto& drhs = reinterpret_cast<CPUDevice*>(rhs.dev)->data;

                auto& d = dlhs * drhs;
                if (lhs.requires_grad) {
                    depends_on.emplace_back(
                            &lhs,
                            [drhs](Eigen::MatrixXd grad) {
                                return Eigen::MatrixXd{grad * drhs.transpose()};
                            });
                }

                if (rhs.requires_grad) {
                    // TODO: those should perform on tensors
                    depends_on.emplace_back(
                            &const_cast<decltype(lhs)>(rhs),
                            [dlhs](Eigen::MatrixXd grad){
                                return Eigen::MatrixXd{dlhs.transpose() * grad};
                            });
                }

                return Tensor{lhs.requires_grad || rhs.requires_grad,
                              depends_on,
                              new CPUDevice{d}};
            }
        };

        Device* dev;

        inline Tensor() = default;

        // TODO: this makes copy for now
        inline Tensor(bool requires_grad = false,
                      const std::vector<Function>& depends_on = std::vector<Function>{},
                      Device* dev = reinterpret_cast<Device *>(new CPUDevice{})) : \
                         depends_on{depends_on},
                         dev{dev}, requires_grad{requires_grad} {
            if (requires_grad) {
                zero_grad();
            }
        };

        inline Tensor& T() {
            this->dev->T();
            return *this;
        }

        // TODO : make this one a general one
        friend inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
            return os << reinterpret_cast<CPUDevice*>(t.dev)->data;
        }

        inline void zero_grad() {
            dev->zero_grad();
        }

        inline void backward(Tensor* grad = nullptr) {
            if (dev->grad == nullptr) {
                dev->grad = Eigen::MatrixXd().setOnes(dev->data.rows(), dev->data.cols());
            }

            if (dev->grad == nullptr) {
                std::cerr << "Could not perform backward on a tensor that does not require gradient" << std::endl;
                exit(1);
            }


            // TODO: add method add for two matrices
            dev->add(dev->grad, grad);

            for (auto &dep : depends_on) {
                func f{dep.op};
                auto gr = grad->dev->data;
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