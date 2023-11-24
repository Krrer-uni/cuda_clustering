//
// Created by krrer on 28.10.23.
//

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "test_interface.cuh"

pcl::PointCloud<pcl::PointXYZ>::Ptr generatePcGrid(size_t block_height,
                                                   size_t block_width,
                                                   size_t grid_height,
                                                   size_t grid_width,
                                                   float block_dist,
                                                   float grid_dist) {
  auto gridPC = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  float x = 0;
  float y = 0;
  for (int g_h = 0; g_h < grid_height; g_h++) {
    for (int b_h = 0; b_h < block_height; b_h++) {
      x = 0;
      for (int g_w = 0; g_w < grid_width; g_w++) {
        for (int b_w = 0; b_w < block_width; b_w++) {
          gridPC->emplace_back(pcl::PointXYZ{x, y, 0.0});
          x += block_dist;
        }
        x += grid_dist;
      }
      y += block_dist;
    }
    y += grid_dist;
  }

  return gridPC;
}

int main(){
 test_cuda_class();
}