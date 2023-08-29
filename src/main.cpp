#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "cuda_clustering.cuh"

int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_data/test_pc.pcd", *cloud) == -1){
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  std::cout << cloud->size() << std::endl;
  cuda_wrapper(5);
  return 0;
}
