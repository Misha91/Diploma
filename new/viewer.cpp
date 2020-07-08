#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

int user_data;

void viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor (0.0, 0.5, 0.0);
    pcl::PointXYZ o;
    o.x = 1.0;
    o.y = 0;
    o.z = 0;
    viewer.addSphere (o, 0.25, "sphere", 0);
    std::cout << "i only run once" << std::endl;

}

void viewerPsycho (pcl::visualization::PCLVisualizer& viewer)
{
    static unsigned count = 0;
    std::stringstream ss;
    ss << "Once per viewer loop: " << count++;
    viewer.removeShape ("text", 0);
    viewer.addText (ss.str(), 200, 300, "text", 0);

    //FIXME: possible race condition here:
    user_data++;
}

int main (int argc, char** argv)
{
  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
  pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());

  //if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd.pcd", *cloud) == -1) //* load the file
  //{
  //  PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
  //  return (-1);
  //}
  pcl::PCDReader reader;
  if (argc > 1)
  {
    reader.read (argv[1], *cloud);
  }
  else
  {
    reader.read ("test_pcd.pcd", *cloud);
  }


  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;

  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (0.02f, 0.02f, 0.02f);
  sor.filter (*cloud_filtered);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_new(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(*cloud_filtered, *cloud_filtered_new);
  std::cerr << "Point cloud data: " << cloud_filtered_new->points.size () << " points" << std::endl;

  /*for (std::size_t i = 0; i < cloud->points.size (); ++i)
    std::cerr << "    " << cloud->points[i].x << " "
                        << cloud->points[i].y << " "
                        << cloud->points[i].z << std::endl;
  */
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  seg.setInputCloud (cloud_filtered_new);
  seg.segment (*inliers, *coefficients);


  pcl::visualization::CloudViewer viewer("Cloud Viewer");

     //blocks until the cloud is actually rendered

  viewer.showCloud(cloud_filtered_new);

  //use the following functions to get access to the underlying more advanced/powerful
  //PCLVisualizer

  //This will only get called once
  viewer.runOnVisualizationThreadOnce (viewerOneOff);

  //This will get called once per visualization iteration
  viewer.runOnVisualizationThread (viewerPsycho);
  while (!viewer.wasStopped ())
  {
  //you can also do cool processing here
  //FIXME: Note that this is running in a separate thread from viewerPsycho
  //and you should guard against race conditions yourself...
  user_data++;
  }

  return (0);
}
