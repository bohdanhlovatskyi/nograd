#ifndef NG_TENSOR_2__
#define NG_TENSOR_2__

#include <Eigen/Dense>

#include <type_traits>
#include <vector>

namespace ng {

    template<typename tensor> struct Function;

    struct tensor {
        struct tensor_traits {
            using data_value_type = void*;
            using grad_fn = std::function<void(void)>;
        };

        using dvt = tensor_traits::data_value_type;

        dvt data;
        dvt grad;
        std::vector<Function<tensor>> depends_on;
        bool requires_grad;

        virtual void T() = 0;
        virtual void zero_grad() = 0;
        virtual void setGradOnes() = 0;
        virtual void print_myself(std::ostream& os) const = 0;
        virtual tensor* matmul(tensor* lhs, const tensor* rhs) = 0;
        static dvt add(dvt& lhs, const dvt& rhs);
    };

    template<typename tensor>
    struct Function {
        using func = typename tensor::tensor_traits::grad_fn;
        // operation that was performed on a ctx
        // instantiation of a function is a context
        tensor* ctx;
        func op;

        Function(tensor* ctx, func op) : ctx{ctx}, op{op} {};
    };

    struct CPUTensor : public tensor {
        struct tensor_traits {
            using data_value_type = Eigen::MatrixXd;
            using grad_fn = std::function<Eigen::MatrixXd(Eigen::MatrixXd)>;
        };

        using dvt = tensor_traits::data_value_type;

        dvt data;
        dvt grad;
        std::vector<Function<CPUTensor>> depends_on;
        bool requires_grad;

        CPUTensor() = default;
        CPUTensor(dvt d) : data{d} {};
        CPUTensor(dvt d, bool requires_grad,
                  dvt grad = dvt{}, std::vector<Function<CPUTensor>> depends_on = {}):\
                  data{d}, grad{grad}, depends_on{depends_on},
                  requires_grad{requires_grad} {};

        inline dvt& getData() {
            return data;
        }

        void print_myself(std::ostream& os) const override {
            os << data << std::endl;
        }

        inline void zero_grad() override {
            grad = Eigen::MatrixXd().setZero(data.rows(), data.cols());
        }

        inline void setGradOnes() override {
            grad = Eigen::MatrixXd().setOnes(data.rows(), data.cols());
        }

        inline void T() override {
            auto& dlhs = reinterpret_cast<CPUTensor*>(this)->data;
            dlhs = dlhs.transpose();
        }

        tensor* matmul(tensor* lhs, const tensor* rhs) override {
            std::vector<Function<CPUTensor>> depends_on;
            depends_on.reserve(lhs->requires_grad + rhs->requires_grad);

            auto& dlhs = dynamic_cast<CPUTensor *>(lhs)->data;
            auto& drhs = dynamic_cast<CPUTensor *>(const_cast<tensor *>(rhs))->data;

            auto& d = dlhs * drhs;
            if (lhs->requires_grad) {
                depends_on.emplace_back(
                        dynamic_cast<CPUTensor*>(lhs),
                        [drhs](Eigen::MatrixXd grad) {
                            return Eigen::MatrixXd{grad * drhs.transpose()};
                        });
            }

            if (rhs->requires_grad) {
                depends_on.emplace_back(
                        dynamic_cast<CPUTensor*>(const_cast<decltype(lhs)>(rhs)),
                        [dlhs](Eigen::MatrixXd grad){
                            return Eigen::MatrixXd{dlhs.transpose() * grad};
                        });
            }

            return new CPUTensor{Eigen::MatrixXd{d}, lhs->requires_grad || rhs->requires_grad,
                                 Eigen::MatrixXd{}, depends_on};
        }

    };

    class Tensor {
    public:
        tensor* tnsr;

        inline Tensor() = default;

        inline Tensor(Eigen::MatrixXd d, bool requires_grad): tnsr{new CPUTensor{d, requires_grad}} {}

        inline Tensor(tensor* tnsr = new CPUTensor{}) {
            if (tnsr->requires_grad) {
                tnsr->zero_grad();
            }
        };

        inline decltype(auto) grad() {
            return tnsr->grad;
        }

        inline decltype(auto) data() {
            return tnsr->data;
        }

        inline Tensor& T() {
            tnsr->T();
            return *this;
        }

        friend inline std::ostream& operator<<(std::ostream& os, const Tensor* t) {
            t->tnsr->print_myself(os);
            return os;
        }

        inline void zero_grad() {
            tnsr->zero_grad();
        }

//        void backward(tensor* upstream_grad = nullptr) {
//            {
//                if (upstream_grad->data == decltype(upstream_grad->data){}) {
//                    upstream_grad->setGradOnes();
//                }
//
//                if (tnsr->grad == nullptr) {
//                    std::cerr << "Could not perform backward on a tensor that does not require gradient" << std::endl;
//                    exit(1);
//                }
//
//                tnsr->grad->data = tnsr->grad->data + grad->grad;
//
//                for (Function<tensor*> dep : grad->depends_on) {
//                    tensor::tensor_traits::grad_fn f{dep.op};
//                    auto gr = grad->data;
//                    auto r = f(gr);
//                    dep->ctx->backward(new decltype(*grad){r});
//                }
//            }
//        }

//        inline void backward() {
//            tnsr->backward();
//        }

        template<typename Tn>
        friend inline Tensor* operator*(Tn& lhs, const Tn& rhs)
        {
            return new Tensor{lhs.tnsr->matmul(lhs.tnsr, rhs.tnsr)};
        }

    };

};

#endif // NG_TENSOR_2__