#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "main_algorithm.h"

extern void detect_planes(pcl::PCLPointCloud2::Ptr cloud_blob);
static int counter = 0;

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  #ifdef DEBUG
  ROS_INFO("\nI see points!\n");
  #endif

  pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
  pcl::PCLPointCloud2::Ptr cloudPTR (cloud);

  pcl_conversions::toPCL(*input, *cloud);
  detect_planes(cloudPTR);
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "listener");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("points", 1000, cloud_cb);

  ros::spin();

  return 0;
}
