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
        tensor* grad;
        std::vector<Function<tensor>> depends_on;
        bool requires_grad;

        virtual void T() = 0;
        virtual void zero_grad() = 0;
        virtual void setGradOnes() = 0;
        virtual void print_myself(std::ostream& os) const = 0;

        virtual void backward(tensor* upstream_grad = nullptr) = 0;

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
        CPUTensor* grad;
        std::vector<Function<CPUTensor>> depends_on;
        bool requires_grad;

        CPUTensor() = default;
        CPUTensor(dvt d) : data{d} {};
        CPUTensor(dvt d, bool requires_grad,
                  CPUTensor* grad = nullptr, std::vector<Function<CPUTensor>> depends_on = {}):\
                  data{d}, requires_grad{requires_grad}, grad{grad}, depends_on{depends_on} {
            if (requires_grad) {
                zero_grad();
            }
        };

        inline dvt& getData() {
            return data;
        }

        void print_myself(std::ostream& os) const override {
            os << data << std::endl;
        }

        inline void zero_grad() override {
            grad = new CPUTensor(Eigen::MatrixXd{data.rows(), data.cols()});
        }

        inline void setGradOnes() override {
            grad = new CPUTensor(Eigen::MatrixXd{}.setOnes(data.rows(), data.cols()));
        }

        void backward(tensor* upstream_grad = nullptr) override {
            {
                if (upstream_grad == nullptr) {
                    upstream_grad = new CPUTensor(
                            Eigen::MatrixXd().setOnes(data.rows(), data.cols())
                    );
                }

                if (requires_grad == false) {
                    std::cerr << "Could not perform backward on a tensor that does not require gradient" << std::endl;
                    exit(1);
                }

                grad->data += dynamic_cast<CPUTensor*>(upstream_grad)->data;

                for (Function<CPUTensor> dep : depends_on) {
                    auto r = dep.op(grad->data);
                    dep.ctx->backward(new CPUTensor{r});
                }
            }
        }

        inline void T() override {
            auto& dlhs = reinterpret_cast<CPUTensor*>(this)->data;
            dlhs = dlhs.transpose();
        }

        tensor* matmul(tensor* lhs, const tensor* rhs) override {
            // without this one it won't store require_grad for some reason
            auto l = dynamic_cast<CPUTensor*>(lhs);
            auto r = dynamic_cast<CPUTensor*>(const_cast<tensor*>(rhs));

            std::vector<Function<CPUTensor>> depends_on;
            depends_on.reserve(l->requires_grad + r->requires_grad);

            auto& ldata = l->data;
            auto& rdata = r->data;

            auto& d = ldata * rdata;
            if (l->requires_grad) {
                depends_on.emplace_back(
                        l,
                        [rdata](Eigen::MatrixXd grad) {
                            return Eigen::MatrixXd{grad * rdata.transpose()};
                        });
            }

            if (r->requires_grad) {
                depends_on.emplace_back(
                        r,
                        [ldata](Eigen::MatrixXd grad){
                            return Eigen::MatrixXd{ldata.transpose() * grad};
                        });
            }

            return new CPUTensor{Eigen::MatrixXd{d}, l->requires_grad || r->requires_grad,
                                 nullptr, depends_on};
        }

        inline friend CPUTensor& operator*(CPUTensor& lhs, const CPUTensor& rhs) {
            return *dynamic_cast<CPUTensor *>(lhs.matmul(&lhs, &rhs));
        }

        inline CPUTensor& relu() {
            auto mask = data.unaryExpr([](double x){ return x > 0 ? 1 : 0; });
            std::vector<Function<CPUTensor>> depends_on;
            depends_on.reserve(requires_grad);

            if (requires_grad) {
                depends_on.emplace_back(
                        this,
                        [](Eigen::MatrixXd grad) {
                            return Eigen::MatrixXd{};
                        });
            }
        }

    };
};

#endif // NG_TENSOR_2__