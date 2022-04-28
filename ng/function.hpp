#ifndef NG_FUNCTION__
#define NG_FUNCTION__

#include <Eigen/Dense>

namespace ng {

    template<typename Func>
    class Function {
    private:
        Func op;
        Tensor* tensor;
    public:
        Function(Tensor* tensor, Func op): tensor{tensor}, op{op} {}
    };
}


#endif // NG_FUNCTION__