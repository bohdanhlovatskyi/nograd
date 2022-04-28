#ifndef NG_FUNCTION__
#define NG_FUNCTION__

namespace ng {

    class Function {

        Function() {}

        /*
         * Accepts context followed by any number of Tensors
         * Context will store the values that will be needed
         * during the backward pass via ctx.save_for_backward(...);
         */
        template<typename Args...>
        virtual static void forward(Context& ctx, args Args...) = 0;

        template<typename Args...>
        virtual static void backward(Context& ctx, args Args...) = 0;
    };


    class Exp : public Function {

        template<typename Args...>
        static void forward(Context& ctx, args Args...) override {
            ;
        }
    };
}


#endif // NG_FUNCTION__