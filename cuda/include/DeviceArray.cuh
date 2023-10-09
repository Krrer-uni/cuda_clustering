#pragma once
#include <cstddef>
#include <vector>

template<typename T>
class DeviceArray {
 public:
  std::size_t size;
  T *data;

};
