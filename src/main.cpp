#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "include/CudaClustering.cuh"
#include <pcl/visualization/pcl_visualizer.h>
#include <chrono>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

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

int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ>("test_data/ransac_test.pcd", *cloud) == -1) {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  cloud = generatePcGrid(11, 11, 100 ,30, 1.0, 2.0);
  std::vector<unsigned> labels(cloud->size());

  auto start = std::chrono::steady_clock::now();
  std::cout << cloud->size() << std::endl;
  CudaClustering<pcl::PointXYZ> clustering;
  clustering.setInputCloud(cloud);
  clustering.setParams({1.3f});
  clustering.extract(labels);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Time(ms) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

  start = std::chrono::steady_clock::now();
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud);
  ec.extract (cluster_indices);
  end = std::chrono::steady_clock::now();
  std::cout << "Time(ms) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

  pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  for (int i = 0; i < cloud->size(); i++) {
    pcl::PointXYZI p;
    p.x = (*cloud)[i].x;
    p.y = (*cloud)[i].y;
    p.z = (*cloud)[i].z;
    p.intensity = labels[i];
    out_cloud->push_back(p);
  }
  pcl::io::savePCDFile("test_data/test_output.pcd", *out_cloud);
  pcl::visualization::PCLVisualizer viewer("Test output");
  pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color("intensity");
  color.setInputCloud(out_cloud);
  viewer.addPointCloud<pcl::PointXYZI>(out_cloud,color, "test_out");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 20, "test_out");
  viewer.spin();
  return 0;
}
