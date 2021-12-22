#pragma once

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/tensor.h"
#include "megdnn/oprs.h"

#include "helper/helper.h"

using namespace mgb;

class LeNet {
protected:
  std::shared_ptr<cg::ComputingGraph> m_graph;
  CompNode m_comp_node;

  std::vector<std::shared_ptr<DeviceTensorND>> m_pre_grads;
  std::vector<std::shared_ptr<DeviceTensorND>> m_weights;

  //   // Input data shape and label shape
  //   TensorShape input_shape = {(unsigned long)batchsize, 1, 28, 28};
  //   TensorShape label_shape = {(unsigned long)batchsize};

  // Shapes of weights
  TensorShape conv1_shape = {6, 1, 5, 5};
  TensorShape conv2_shape = {16, 6, 5, 5};
  TensorShape conv1_bias_shape = {1, 6, 1, 1};
  TensorShape conv2_bias_shape = {1, 16, 1, 1};
  TensorShape fc1_shape = {16 * 5 * 5, 120};
  TensorShape fc2_shape = {120, 84};
  TensorShape fc3_shape = {84, 10};
  TensorShape fc1_bias_shape = {120};
  TensorShape fc2_bias_shape = {84};
  TensorShape fc3_bias_shape = {10};

  SymbolVarArray symbol_weights;

public:
  LeNet(std::shared_ptr<cg::ComputingGraph> graph, CompNode comp_node);

  SymbolVar build(std::shared_ptr<HostTensorND> data,
                  std::shared_ptr<HostTensorND> label);

  const SymbolVarArray &get_weights() { return symbol_weights; }
};
