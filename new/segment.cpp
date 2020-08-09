#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

int
main (int argc, char** argv)
{
  pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2), cloud_filtered_blob (new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the cloud data
  pcl::PCDReader reader;
  if (argc > 1)
  {
    reader.read (argv[1], *cloud_blob);
  }
  else
  {
    reader.read ("test_pcd.pcd", *cloud_blob);
  }

  std::cerr << "PointCloud before filtering: " << cloud_blob->width * cloud_blob->height << " data points." << std::endl;

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud_blob);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered_blob);

  // Convert to the templated PointCloud
  pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);

  std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;

  // Write the downsampled version to disk
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ> ("table_scene_lms400_downsampled.pcd", *cloud_filtered, false);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (10000);
  seg.setDistanceThreshold (0.075); //0.075

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;

  int i = 0, nr_points = (int) cloud_filtered->points.size ();
  // While 30% of the original cloud is still there
  pcl::PointCloud<pcl::PointXYZRGB> output_cloud;
  while (cloud_filtered->points.size () > 0.01 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the inliers
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_p);
    std::cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;
    int size_base = output_cloud.size();
    output_cloud.points.resize(size_base + cloud_p->width * cloud_p->height);

    for (int q=0; q < (cloud_p->width * cloud_p->height); q++)
    {
      output_cloud.points[size_base + q].x = cloud_p->points[q].x;
      output_cloud.points[size_base + q].y = cloud_p->points[q].y;
      output_cloud.points[size_base + q].z = cloud_p->points[q].z;
      #define uint8_t unsigned char
      #define uint32_t unsigned int
      uint8_t r = 125*(uint8_t)(i%3 == 0) + 25*i , g = 125*(uint8_t)(i%3 == 1) + 25*i, b = 125*(uint8_t)(i%3 == 2) + 25*i;    // Example: Red color
      uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
      output_cloud.points[size_base + q].rgb = *reinterpret_cast<float*>(&rgb);
    }



    std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                                     << coefficients->values[1] << " "
                                     << coefficients->values[2] << " "
                                     << coefficients->values[3] << std::endl;
   std::stringstream ss1;
   ss1 << "PLANE_SEGM_" <<  i << ".pcd";
   writer.write<pcl::PointXYZ> (ss1.str (), *cloud_p, false);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);
    i++;
  }
  output_cloud.width = output_cloud.points.size();
  output_cloud.height = 1;
  std::stringstream ss;
  ss << "PLANE_SEGM" << ".pcd";
  writer.write<pcl::PointXYZRGB> (ss.str (), output_cloud, false);
  return (0);
}
