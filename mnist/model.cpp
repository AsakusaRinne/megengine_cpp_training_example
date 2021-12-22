#include "mnist/model.h"

#include <cassert>

#define gen_float32_gaussian                                                   \
  HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN>
#define gen_float32_constant                                                   \
  HostTensorGenerator<dtype::Float32, RandomDistribution::CONSTANT>
#define gen_int32                                                              \
  HostTensorGenerator<dtype::Int32, RandomDistribution::CONSTANT>

/*
 * Convert the tensor from host to device.
 */
std::shared_ptr<DeviceTensorND>
get_device_tensor(std::shared_ptr<HostTensorND> inp) {
  std::shared_ptr<DeviceTensorND> r =
      std::make_shared<DeviceTensorND>(inp->comp_node(), inp->layout());
  auto device_ptr = r->raw_ptr();
  auto host_ptr = inp->raw_ptr();
  for (size_t i = 0; i < inp->layout().access_bytes(); i++) {
    device_ptr[i] = host_ptr[i];
  }
  return r;
}

LeNet::LeNet(std::shared_ptr<cg::ComputingGraph> graph,
             CompNode comp_node = CompNode::load("xpu0")) {
  m_comp_node = comp_node;
  if (!m_comp_node.valid()) {
    m_comp_node = CompNode::load("xpu0");
  }

  m_graph = graph;

  // conv1
  m_weights.push_back(get_device_tensor(gen_float32_gaussian(
      0.0f, std::sqrt(1 / (5.0f * 5.0f)))(conv1_shape, m_comp_node)));
  // conv2
  m_weights.push_back(get_device_tensor(gen_float32_gaussian(
      0.0f, std::sqrt(1 / (6.0f * 5.0f * 5.0f)))(conv2_shape, m_comp_node)));
  // conv1_bias
  m_weights.push_back(get_device_tensor(gen_float32_gaussian(
      0.0f, std::sqrt(1 / 6.0f))(conv1_bias_shape, m_comp_node)));
  // conv2_bias
  m_weights.push_back(get_device_tensor(gen_float32_gaussian(
      0.0f, std::sqrt(1 / 16.0f))(conv2_bias_shape, m_comp_node)));
  // fc1
  m_weights.push_back(get_device_tensor(gen_float32_gaussian(
      0.0f, std::sqrt(1 / (16.0f * 5.0f * 5.0f)))(fc1_shape, m_comp_node)));
  // fc2
  m_weights.push_back(get_device_tensor(gen_float32_gaussian(
      0.0f, std::sqrt(1 / 120.0f))(fc2_shape, m_comp_node)));
  // fc3
  m_weights.push_back(get_device_tensor(gen_float32_gaussian(
      0.0f, std::sqrt(1 / 84.0f))(fc3_shape, m_comp_node)));
  // fc1_bias
  m_weights.push_back(get_device_tensor(
      gen_float32_gaussian(0.0f, std::sqrt(1 / (16.0f * 5.0f * 5.0f)))(
          fc1_bias_shape, m_comp_node)));
  // fc2_bias
  m_weights.push_back(get_device_tensor(gen_float32_gaussian(
      0.0f, std::sqrt(1 / 120.0f))(fc2_bias_shape, m_comp_node)));
  // fc3_bias
  m_weights.push_back(get_device_tensor(gen_float32_gaussian(
      0.0f, std::sqrt(1 / 84.0f))(fc3_bias_shape, m_comp_node)));

  m_pre_grads.push_back(
      get_device_tensor(gen_float32_constant(.0f)(conv1_shape, m_comp_node)));
  m_pre_grads.push_back(
      get_device_tensor(gen_float32_constant(.0f)(conv2_shape, m_comp_node)));
  m_pre_grads.push_back(get_device_tensor(
      gen_float32_constant(.0f)(conv1_bias_shape, m_comp_node)));
  m_pre_grads.push_back(get_device_tensor(
      gen_float32_constant(.0f)(conv2_bias_shape, m_comp_node)));
  m_pre_grads.push_back(
      get_device_tensor(gen_float32_constant(.0f)(fc1_shape, m_comp_node)));
  m_pre_grads.push_back(
      get_device_tensor(gen_float32_constant(.0f)(fc2_shape, m_comp_node)));
  m_pre_grads.push_back(
      get_device_tensor(gen_float32_constant(.0f)(fc3_shape, m_comp_node)));
  m_pre_grads.push_back(get_device_tensor(
      gen_float32_constant(.0f)(fc1_bias_shape, m_comp_node)));
  m_pre_grads.push_back(get_device_tensor(
      gen_float32_constant(.0f)(fc2_bias_shape, m_comp_node)));
  m_pre_grads.push_back(get_device_tensor(
      gen_float32_constant(.0f)(fc3_bias_shape, m_comp_node)));
}

SymbolVar LeNet::build(std::shared_ptr<HostTensorND> data,
                       std::shared_ptr<HostTensorND> label) {
  *data = *gen_float32_gaussian()(data->shape(), m_comp_node);
  *label = *gen_int32(-1)(label->shape(), m_comp_node);

  assert(data->shape().ndim == 4);
  assert(label->shape().ndim == 1);
  unsigned long batchsize = data->shape()[0];

  // set params
  opr::Pooling::Param pooling_param(megdnn::Pooling::Param::Mode::MAX, 0, 0, 2,
                                    2, 2, 2);
  pooling_param.format = opr::Pooling::Param::Format::NCHW;
  megdnn::Convolution::Param conv_param;
  conv_param.mode = megdnn::Convolution::Param::Mode::CROSS_CORRELATION;

  megdnn::Padding::Param padding_param;
  padding_param.padding_mode = megdnn::Padding::Param::PaddingMode::CONSTANT;
  padding_param.padding_val = .0f;
  padding_param.front_offset_dim2 = 2;
  padding_param.front_offset_dim3 = 2;
  padding_param.back_offset_dim2 = 2;
  padding_param.back_offset_dim3 = 2;

  for (auto m_weight : m_weights) {
    symbol_weights.push_back(opr::SharedDeviceTensor::make(*m_graph, m_weight));
  }

  // define symbolvars
  // first conv layer
  SymbolVar symbol_input = opr::Padding::make(
      opr::Host2DeviceCopy::make(*m_graph, data), padding_param);
  //   SymbolVar symbol_label = opr::Host2DeviceCopy::make(*m_graph, label);

  SymbolVar symbol_conv1 =
      opr::Convolution::make(symbol_input, symbol_weights[0], conv_param);
  symbol_conv1 = opr::relu(symbol_conv1 + symbol_weights[2]);
  SymbolVar symbol_maxpool1 = opr::Pooling::make(symbol_conv1, pooling_param);

  // second conv layer and flatten
  SymbolVar symbol_conv2 =
      opr::Convolution::make(symbol_maxpool1, symbol_weights[1], conv_param);
  symbol_conv2 = opr::relu(symbol_conv2 + symbol_weights[3]);
  SymbolVar symbol_maxpool2 =
      opr::Pooling::make(symbol_conv2, pooling_param)
          .reshape({(unsigned long)batchsize, fc1_shape[0]});

  // first fc layer
  SymbolVar symbol_fc1 =
      opr::MatrixMul::make(symbol_maxpool2, symbol_weights[4]) +
      symbol_weights[7];
  symbol_fc1 = opr::relu(symbol_fc1);

  // second fc layer
  SymbolVar symbol_fc2 =
      opr::MatrixMul::make(symbol_fc1, symbol_weights[5]) + symbol_weights[8];
  symbol_fc2 = opr::relu(symbol_fc2);

  // third fc layer(output layer)
  SymbolVar symbol_fc3 =
      opr::MatrixMul::make(symbol_fc2, symbol_weights[6]) + symbol_weights[9];

  //   SymbolVar symbol_output = opr::Argmax::make(symbol_fc3, {1});

  return symbol_fc3;
}
