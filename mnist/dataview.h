#pragma once

#include "megbrain/opr/training/dataview.h"
#include "megbrain/tensor.h"

#include <cassert>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace mgb;

float normalize(float inp, float mean, float std) { return (inp - mean) / std; }

float transform(float inp) {
  return normalize(inp, 0.1307f * 255, 0.3081f * 255);
}

/*
 * Define the dataset special for Mnist.
 */
class MnistDataset : public IDataView {
public:
  enum Mode : uint32_t { TRAIN = 0, TEST = 1 };

  MnistDataset(Mode mode, std::string dir_name);
  void load_data(Mode mode, std::string dir_name);
  DataPair get_item(int idx);
  size_t size();

protected:
  const std::string TRAIN_IMAGE_FILENAME = "train-images-idx3-ubyte";
  const std::string TRAIN_LABEL_FILENAME = "train-labels-idx1-ubyte";
  const std::string TEST_IMAGE_FILENAME = "t10k-images-idx3-ubyte";
  const std::string TEST_LABEL_FILENAME = "t10k-labels-idx1-ubyte";

  std::vector<DataPair> dataset;
};

/*
 * Since the structure of a byte is different on different devices, we need
 to
 * reverse the data if needed.
 */
int ReverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

MnistDataset::MnistDataset(Mode mode, std::string dir_name) {
  load_data(mode, dir_name);
}

/*
 * The method to load data from the files of mnist dataset.
 */
void MnistDataset::load_data(Mode mode, std::string dir_name) {
  std::ifstream ifs_image;
  std::ifstream ifs_label;

  // open the train images stream and train labels stream
  std::string filename_image, filename_label;
  if (dir_name.at(dir_name.size() - 1) == '/') {
    filename_image = dir_name + (mode == Mode::TRAIN ? TRAIN_IMAGE_FILENAME
                                                     : TEST_IMAGE_FILENAME);
    filename_label = dir_name + (mode == Mode::TRAIN ? TRAIN_LABEL_FILENAME
                                                     : TEST_LABEL_FILENAME);
  } else {
    filename_image =
        dir_name + "/" +
        (mode == Mode::TRAIN ? TRAIN_IMAGE_FILENAME : TEST_IMAGE_FILENAME);
    filename_label =
        dir_name + "/" +
        (mode == Mode::TRAIN ? TRAIN_LABEL_FILENAME : TEST_LABEL_FILENAME);
  }
  ifs_image.open(filename_image, std::ios::in);
  assert(ifs_image.is_open());
  ifs_label.open(filename_label, std::ios::in);
  assert(ifs_label.is_open());

  // read the header of train image file
  int magic_number;
  ifs_image.read((char *)&magic_number, sizeof(magic_number));
  bool need_reverse = magic_number != 2051;

  int image_count;
  ifs_image.read((char *)&image_count, sizeof(image_count));
  image_count = need_reverse ? ReverseInt(image_count) : image_count;

  uint32_t row_nums, col_nums;
  ifs_image.read((char *)&row_nums, sizeof(row_nums));
  ifs_image.read((char *)&col_nums, sizeof(col_nums));
  row_nums = need_reverse ? ReverseInt(row_nums) : row_nums;
  col_nums = need_reverse ? ReverseInt(col_nums) : col_nums;
  TensorShape image_shape = {1, row_nums, col_nums};
  TensorShape label_shape = {1};

  // read the header of train label file
  ifs_label.read((char *)&magic_number, sizeof(magic_number));
  assert((need_reverse ? ReverseInt(magic_number) : magic_number) == 2049);

  int label_count;
  ifs_label.read((char *)&label_count, sizeof(label_count));
  label_count = need_reverse ? ReverseInt(label_count) : label_count;
  assert(image_count == label_count);

  for (int i = 0; i < image_count; i++) {
    auto image = std::make_shared<HostTensorND>(CompNode::load("xpu0"),
                                                image_shape, dtype::Float32());
    auto label = std::make_shared<HostTensorND>(CompNode::load("xpu0"),
                                                label_shape, dtype::Int32());
    // read the image pixel by pixel
    auto image_ptr = image->ptr<float>();
    for (unsigned int j = 0; j < row_nums * col_nums; j++) {
      uint8_t file_value;
      ifs_image.read((char *)&file_value, sizeof(file_value));
      image_ptr[j] = transform(file_value);
    }
    // read the label
    uint8_t file_value;
    ifs_label.read((char *)&file_value, sizeof(file_value));
    auto label_ptr = label->ptr<int>();
    label_ptr[0] = file_value;
    dataset.push_back({image, label});
  }
}

DataPair MnistDataset::get_item(int idx) { return dataset.at(idx); }

size_t MnistDataset::size() { return dataset.size(); }