
#include "mainfile/train.h"
#include "megbrain/opr/training/loss.h"
#include "megbrain/opr/training/optimizer.h"
#include "mnist/dataview.h"
#include "mnist/model.h"
#include "progressbar/include/progressbar.hpp"

#include <iostream>

using namespace mgb;
using namespace loss;
using namespace optimizer;

// the shape of the image
static const unsigned long channels = 1;
static const unsigned long height = 28;
static const unsigned long width = 28;

// the path of the dataset
static std::string dataset_dir = "mnist/dataset";

// the vector to get preditions
std::vector<int> pred;

// the data and label used for training and test
std::shared_ptr<HostTensorND> data;
std::shared_ptr<HostTensorND> label;

/*
 * Get the labels from the batched label tensor
 */
std::vector<int> get_labels(std::shared_ptr<HostTensorND> data) {
  std::vector<int> r;
  // data->sync();
  auto ptr = data->ptr<int>();
  for (size_t i = 0, it = data->shape().total_nr_elems(); i < it; i += 1) {
    r.push_back(ptr[i]);
  }
  return r;
}

// The call back for prediction
void PredictionCallback(DeviceTensorND &data) {
  int *ptr;
  if (data.comp_node().device_type() == CompNode::DeviceType::CPU) {
    data.sync();
    ptr = data.ptr<int>();
  } else {
    HostTensorND m_host;
    m_host.copy_from(data).sync();
    ptr = m_host.ptr<int>();
  }
  for (size_t i = 0, it = data.shape().total_nr_elems(); i < it; i += 1) {
    pred.at(i) = ptr[i];
  }
};

void test(std::shared_ptr<ComputingGraph> graph, CompNode comp_node,
          SymbolVar symbol_oup, unsigned long batchsize) {
  // declare the test dataset
  auto test_dataset =
      std::make_shared<MnistDataset>(MnistDataset::Mode::TEST, dataset_dir);
  DataLoader test_dataloader(test_dataset, comp_node, batchsize, false, true);

  // get the result symbolvar and get the executing sequence
  SymbolVar symbol_result = opr::Argmax::make(symbol_oup, {1});
  cg::ComputingGraph::OutputSpec spec;
  spec.push_back({symbol_result, PredictionCallback});
  auto func = graph->compile(spec);

  // initialize the prediction vector
  for (unsigned int i = 0; i < batchsize; i++) {
    pred.push_back(-1);
  }

  std::cout << std::endl << "start testing..." << std::endl;
  progressbar bar(test_dataloader.size());
  int correct_num = 0;
  for (size_t i = 0; i < test_dataloader.size(); i++) {
    auto item = test_dataloader.next();
    data->copy_from(*(item.first)).sync();
    label->copy_from(*(item.second)).sync();

    // run the graph
    func->execute().wait();
    auto results = get_labels(label);
    for (size_t i = 0; i < results.size(); i++) {
      if (std::abs(results.at(i) - pred.at(i)) < 0.01f) {
        correct_num += 1;
      }
    }
    bar.update();
  }

  double acc = static_cast<double>(correct_num) /
               static_cast<double>(test_dataset->size());
  if (acc > 0.95) {
    std::cout << "\nTraining Succeeded!" << std::endl;
  } else {
    std::cout << "\nTraining Failed!" << std::endl;
  }
  std::cout << "There are " << correct_num << " correct predictions of total "
            << test_dataset->size() << " test data.\n The accuracy rate is "
            << acc << std::endl;
}

void train(unsigned long batchsize, DTypeEnum dtype, int epochs) {
  // initialize the graph, computing node and the model
  auto graph = cg::ComputingGraph::make();
  auto comp_node = CompNode::load("xpu0");
  auto model = LeNet(graph, comp_node);

  // specify the input shape with the given batchsize
  data = std::make_shared<HostTensorND>(
      comp_node, TensorShape({batchsize, channels, height, width}),
      DType::from_enum(dtype));
  label = std::make_shared<HostTensorND>(comp_node, TensorShape({batchsize}),
                                         dtype::Int32());

  // declare the dataset
  auto train_dataset =
      std::make_shared<MnistDataset>(MnistDataset::Mode::TRAIN, dataset_dir);
  auto train_dataloader =
      DataLoader(train_dataset, comp_node, batchsize, true, true);

  // get the output symbolval of the model
  auto symbol_oup = model.build(data, label);

  // declare the loss and optimizer used in training
  auto opt = SGD(0.01f, 5e-4f, .9f);
  auto loss = CrossEntropyLoss();

  // complete the computing graph with the loss and optimizer
  // get the loss symbolvar
  SymbolVar symbol_label = opr::Host2DeviceCopy::make(*graph, label);
  SymbolVar symbol_loss = loss(symbol_oup, symbol_label);

  // get the grads
  SymbolVarArray symbol_grads;
  for (auto symbol_weight : model.get_weights()) {
    symbol_grads.push_back(cg::grad(symbol_loss, symbol_weight));
  }

  // update the weights
  SymbolVarArray symbol_updates =
      opt.make_multiple(model.get_weights(), symbol_grads, graph);

  // declare the output spec, which is used to execute the graph
  cg::ComputingGraph::OutputSpec spec;
  for (size_t i = 0; i < symbol_updates.size(); i++) {
    spec.push_back({symbol_updates[i], [&](DeviceTensorND &data) {}});
  }

  // get the executing sequence of the graph and spec
  auto func = graph->compile(spec);

  // start the training
  std::cout << "start training..." << std::endl;
  progressbar bar(epochs * train_dataloader.size());
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (size_t i = 0; i < train_dataloader.size(); i++) {
      auto item = train_dataloader.next();
      // read input data
      data->copy_from(*(item.first)).sync();
      label->copy_from(*(item.second)).sync();

      // run the graph
      func->execute().wait();
      bar.update();
    }
  }

  // test the performance of the model
  test(graph, comp_node, symbol_oup, batchsize);
}

void train_mnist() {
  unsigned long batchsize;

  std::cout << "==========================Training Test of "
               "Mnist=========================="
            << std::endl;
  std::cout << "Press input 0 to skip the test or input a positive number of "
               "batchsize to continue."
            << std::endl;
  std::cin >> batchsize;

  if (!batchsize) {
    return;
  }

  std::cout << "Please input the dtype of training." << std::endl;

  print_dtype_asking_info();

  int dtype_value;
  std::cin >> dtype_value;

  std::cout << "Please input the epochs of training." << std::endl;

  int epochs;
  std::cin >> epochs;
  assert(epochs > 0);

  std::cout << "Please input the path of the dataset." << std::endl;
  std::string s;
  std::cin >> s;
  if (!s.empty()) {
    dataset_dir = s;
  }

  train(batchsize, static_cast<DTypeEnum>(dtype_value), epochs);
}