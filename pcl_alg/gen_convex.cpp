#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <Eigen/Dense>

float calculateAreaPolygon(const pcl::PointCloud<pcl::PointXYZ> &polygon )
{
        float area=0.0;
        int num_points = polygon.size();
  int j = 0;
        Eigen::Vector3f va,vb,res;
        res(0) = res(1) = res(2) = 0.0f;
  for (int i = 0; i < num_points; ++i)
  {
      j = (i + 1) % num_points;
                        va = polygon[i].getVector3fMap();
                        vb = polygon[j].getVector3fMap();
      res += va.cross(vb);
  }
  area=res.norm();
  return area*0.5;
}


int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>),
                                      cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>),
                                      cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCDReader reader;

  //reader.read ("table_scene_mug_stereo_textured.pcd", *cloud);
  if (argc > 1)
  {
    reader.read (argv[1], *cloud);
  }
  else
  {
    reader.read ("test_pcd.pcd", *cloud);
  }
  // Build a filter to remove spurious NaNs
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0, 1.1);
  pass.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering has: "
            << cloud_filtered->points.size () << " data points." << std::endl;


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

  seg.setInputCloud (cloud_filtered);
  seg.segment (*inliers, *coefficients);
  std::cerr << "PointCloud after segmentation has: "
            << inliers->indices.size () << " inliers." << std::endl;

  // Project the model inliers
  pcl::ProjectInliers<pcl::PointXYZ> proj;
  proj.setModelType (pcl::SACMODEL_PLANE);
  // proj.setIndices (inliers);
  proj.setInputCloud (cloud_filtered);
  proj.setModelCoefficients (coefficients);
  proj.filter (*cloud_projected);
  std::cerr << "PointCloud after projection has: "
            << cloud_projected->points.size () << " data points." << std::endl;

  // Create a Concave Hull representation of the projected inliers
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConvexHull<pcl::PointXYZ> chull;
  chull.setInputCloud (cloud_projected);
  //chull.setAlpha (0.1);
  chull.reconstruct (*cloud_hull);


  chull.setComputeAreaVolume(true);
  std::cerr << "Area is " << chull.getTotalArea() << std::endl;
  std::cerr << "Convex hull has: " << cloud_hull->points.size ()
            << " data points." << std::endl;
  std::cerr << "Area is " << calculateAreaPolygon(*cloud_hull) << std::endl;
  pcl::PCDWriter writer;
  writer.write ("table_scene_mug_stereo_textured_hull.pcd", *cloud_hull, false);

  return (0);
}
