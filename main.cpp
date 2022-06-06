// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>

#include <Eigen/Dense>
#include <torch/torch.h>

#include "config_reader/config_reader.hpp"

#include "ng/tensor.hpp"
#include "ng/optim.hpp"
#include "models/mnist_net.h"

void mul_example() {
    Eigen::MatrixXd aa{2,1};
    aa(0,0) = 2.;
    aa(1,0) = 3.;
    std::cout << aa << std::endl;

    Eigen::MatrixXd bb{2,1};
    bb(0,0) = 6.;
    bb(1,0) = 2.;
    std::cout << bb << std::endl;

    auto a = ng::CPUTensor{aa.transpose(), true};
    auto b = ng::CPUTensor{bb, true};

    auto z = a*b;

    std::cout << z.data << " " << z.requires_grad << std::endl;

    z.backward();

    std::cout << (*a.grad).data << std::endl;
    std::cout << (*b.grad).data << std::endl;
}

Eigen::MatrixXd torch_tensor_to_eigen(torch::Tensor& data) {
    float* rd = data.data_ptr<float>();

    // 28 * 28 = 784 * 1
    Eigen::Map<Eigen::MatrixXf> E(rd,
                                  1, data.size(2) * data.size(3));
    return Eigen::MatrixXd{E.cast<double>()};
}

int main(int argc, char* argv[]) {
    std::string filename{"index.cfg"};
    if(argc == 2) {
        filename = argv[1];
    }

    std::ifstream config_stream(filename);
    const auto config = conf(config_stream);

    auto net = new MnistNet{};
    auto optim = new ng::optim::SGD{std::vector<ng::CPUTensor *>{net->l1, net->l2}};

    auto train_dataset = torch::data::datasets::MNIST(config.kDataRoot)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

//    auto test_dataset = torch::data::datasets::MNIST(
//            kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
//            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//            .map(torch::data::transforms::Stack<>());

    auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                    std::move(train_dataset), config.kTrainBatchSize);

//    auto test_loader =
//            torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

    for (auto& td : *train_loader) {
        auto data = torch_tensor_to_eigen(td.data);
        auto target = td.target;
        int cls_id = target[0].item<int>();

        Eigen::MatrixXd e{Eigen::MatrixXd().setZero(1, 10)};
        e(0, cls_id) = 1;

        auto res = net->forward(*(new ng::CPUTensor{data}));
        auto loss = res.mse(*(new ng::CPUTensor{e}));

        std::cout << loss.data << std::endl;

        optim->zero_grad();
        loss.backward();
        optim->step();
    }

    return 0;
}
