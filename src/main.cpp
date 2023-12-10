#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "include/MatrixClustering.cuh"
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
  float x;
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
  MatrixClustering<pcl::PointXYZI> clustering;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<double> times_matrix;
  std::vector<double> times_pcl;
  std::vector<size_t> cloud_size;

  std::vector<std::vector<int>> labels;
  std::vector<pcl::PointIndices> cluster_indices;

  std::cout << sizeof(CudaPoint) << " " << sizeof (pcl::PointXYZI) << std::endl;
  for (int i = 0; i < 2; i++) {
    std::string filename = "test_data/slalom_debug/output_pc_";
    filename += std::to_string(i) + ".pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(filename, *cloud) == -1) {
      PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
      return (-1);
    }
//    cloud = generatePcGrid(10,10,i,i,0.1,0.5);
    cloud_size.push_back(cloud->size());
    labels.resize(cloud->size());
    auto start = std::chrono::high_resolution_clock ::now();
//    std::cout << cloud->size() << std::endl;
    clustering.setInputCloud(cloud);
    clustering.setConfig({0.4f});
    clustering.extract(labels);
    auto end = std::chrono::high_resolution_clock ::now();
//    std::cout << "Time(ms) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
//              << std::endl;
    times_matrix.push_back((end - start)/std::chrono::microseconds(1)/1000.0);

    start = std::chrono::high_resolution_clock::now();
//    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//    tree->setInputCloud(cloud);
//    ec.setClusterTolerance(0.2);
//    ec.setMinClusterSize(0);
//    ec.setMaxClusterSize(25000);
//    ec.setSearchMethod(tree);
//    ec.setInputCloud(cloud);
//    ec.extract(cluster_indices);
    end = std::chrono::high_resolution_clock::now();
//    std::cout << "Time(ms) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
//              << std::endl;
    times_pcl.push_back((end - start)/std::chrono::microseconds(1)/1000.0);

//    std::sort(labels.begin(),labels.end());
//    auto last = std::unique(labels.begin(),labels.end());
//    labels.erase(last,labels.end());
//    assert(cluster_indices.size() == labels.size());
  }

  auto mean_matrix = std::accumulate(times_matrix.begin(), times_matrix.end(), 0.0) / times_matrix.size();
  auto mean_pcl = std::accumulate(times_pcl.begin(), times_pcl.end(), 0.0) / times_pcl.size();
  std::cout << mean_matrix << " " << mean_pcl << std::endl;

  ofstream slalomGpu("dataGpu.txt");
  ofstream slalomPcl("dataPcl.txt");
  ofstream pclSize("cloudSize.txt");

  for(auto time : times_matrix){
    slalomGpu << time << "\n";
  }

  slalomGpu.close();

  for(auto time : times_pcl){
    slalomPcl << time << "\n";
  }
  slalomPcl.close();

  for(auto time : cloud_size){
    pclSize << time << "\n";
  }
  pclSize.close();

  int k = 0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  for (const auto& cluster : labels) {
    for(const auto& i : cluster){
      pcl::PointXYZI p;
      p.x = (*cloud)[i].x;
      p.y = (*cloud)[i].y;
      p.z = (*cloud)[i].z;
      p.intensity = k;
      out_cloud->push_back(p);
    }
    pcl::visualization::PCLVisualizer viewer("Test output");
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color("intensity");
    color.setInputCloud(out_cloud);
    viewer.addPointCloud<pcl::PointXYZI>(out_cloud,color, "test_out");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "test_out");
    viewer.spin();
    k++;
  }
//  pcl::io::savePCDFile("test_data/test_output.pcd", *out_cloud);
//  pcl::visualization::PCLVisualizer viewer("Test output");
//  pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color("intensity");
//  color.setInputCloud(out_cloud);
//  viewer.addPointCloud<pcl::PointXYZI>(out_cloud,color, "test_out");
//  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "test_out");
//  viewer.spin();
  return 0;
}
