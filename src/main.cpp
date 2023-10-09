#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "include/CudaClustering.cuh"
int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_data/ransac_test.pcd", *cloud) == -1){
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  std::vector<unsigned> labels(cloud->size());

  std::cout << cloud->size() << std::endl;
  CudaClustering<pcl::PointXYZ> clustering;
  clustering.setInputCloud(cloud);
  clustering.setParams({0.3f});
  clustering.extract(labels);

  pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  for(int i = 0; i < cloud->size(); i++){
    pcl::PointXYZI p;
    p.x = (*cloud)[i].x;
    p.y = (*cloud)[i].y;
    p.z = (*cloud)[i].z;
    p.intensity = labels[i];
    out_cloud->push_back(p);
  }
  pcl::io::savePCDFile("test_data/test_output.pcd", *out_cloud);
  return 0;
}
