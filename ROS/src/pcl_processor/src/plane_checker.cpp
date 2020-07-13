#include <ros/ros.h>
#include "std_msgs/String.h"
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <time.h>
#include <iostream>

#include "main_algorithm.h"

ros::Publisher chatter_pub;

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  clock_t tStart = clock();
  #ifdef DEBUG
  ROS_INFO("\nI see points!\n");
  #endif
  std::cout << "reseived PCD at" << ros::Time::now() << std::endl;
  static int counter = 0;
  static std::string result;

  pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
  pcl::PCLPointCloud2::Ptr cloudPTR (cloud);

  pcl_conversions::toPCL(*input, *cloud);

  result = detect_planes(cloudPTR, counter);
  std_msgs::String msg;
  std::stringstream ss;
  ss << result << "," << counter;
  msg.data = ss.str();
  chatter_pub.publish(msg);
  std::cout << "published response at" << ros::Time::now() << std::endl;
  counter++;
  printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "listener");
  ros::NodeHandle n;
  chatter_pub = n.advertise<std_msgs::String>("plane_check_result", 1000);
  ros::Subscriber sub = n.subscribe("points", 1000, cloud_cb);


  ros::spin();

  return 0;
}
