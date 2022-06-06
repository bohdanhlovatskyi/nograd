#ifndef OPTIM_H
#define OPTIM_H

#include <vector>
#include "tensor.hpp"

namespace ng {
    namespace optim {
        class Optimizer {
        protected:
            std::vector<ng::CPUTensor *> params;
        public:
            Optimizer() = default;
            Optimizer(std::vector<ng::CPUTensor *> params) : params{params} {};

            void zero_grad() {
                // here param is a pointer
                for (auto param : params) {
                    param->zero_grad();
                }
            }
        };

        class SGD : public Optimizer {
        private:
            double lr;
            double alpha;
            bool reg;
        public:
            inline SGD(std::vector<ng::CPUTensor *> params, double lr = 0.01,
                bool reg = false, double alpha = 0.001) : \
            lr{lr}, alpha{alpha}, reg{reg}
            {
                Optimizer{params};
            }

            void step() {
                for (auto param : params) {
                    param->data -= lr * (param->grad->data + alpha * param->data);
                }
            }
        };
    }
}

#endif // OPTIM_H