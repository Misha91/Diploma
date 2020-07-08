#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/filter.h>

#include <vector>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <queue>
#include <time.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>


#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <Eigen/Dense>

#define SEGM_DIST_THRESHOLD 0.01//0.01 0.1
#define CONV_DIST_THRESHOLD 0.03//0.01
#define MIN_NUM_POINTS_FOR_PLANE 100
#define POINTS_FOR_DIST_CHECK 51 // TO BE ODD!

//supportive funstions - implementation is below main()
pcl::PointCloud<pcl::PointXYZ>::Ptr smooth(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
float calculateAreaPolygon(const pcl::PointCloud<pcl::PointXYZ> &polygon);
float cacl_area(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int j);
std::vector <float> calc_dist_to_plane(std::vector <float> &ground_coeffs, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster);

//main algorithm
void detect_planes(pcl::PCLPointCloud2::Ptr cloud_blob)
{
  //Step 1. Segment all planes
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCLPointCloud2::Ptr cloud_filtered_blob (new pcl::PCLPointCloud2);

  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud_blob);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered_blob);

  // Convert to the templated PointCloud
  pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);

  std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;

  // Create writer
  pcl::PCDWriter writer;


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
  seg.setDistanceThreshold (SEGM_DIST_THRESHOLD);

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  std::vector < pcl::PointCloud<pcl::PointXYZ> > segm_planes;
  std::vector <float> ground_coeffs;
  std::vector <float> dist_vector;

  int i = 0, nr_points = (int) cloud_filtered->points.size ();
  char break_condition = 0;
  // While MIN_NUM_POINTS_FOR_PLANE size cloud is still there
  while (cloud_filtered->points.size () > MIN_NUM_POINTS_FOR_PLANE)
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


    //std::vector<int> ind;
    //pcl::removeNaNFromPointCloud(*cloud_p, *cloud_p, ind);

    //first plane supposed to be a ground - store ground plane coefficients
    if (ground_coeffs.size() == 0)
    {
      ground_coeffs.push_back(coefficients->values[0]);
      ground_coeffs.push_back(coefficients->values[1]);
      ground_coeffs.push_back(coefficients->values[2]);
      ground_coeffs.push_back(coefficients->values[3]);
    }

    else
    {
      dist_vector = calc_dist_to_plane(ground_coeffs, cloud_p);
      if (dist_vector[1] >= 0.075) break_condition = 1;
    }


    //segm_planes.push_back(*cloud_p);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);

    if (break_condition)
    {
      segm_planes.push_back(*cloud_filtered);
      break;
    }
  }

  printf("Found %d planes\n", (int)segm_planes.size());
  // END OF STEP 1



  //STEP 2. CHECK PLANES
  int j = 0, l = 0;
  for (i = 0; i< segm_planes.size() ; i++)
  {
    printf("Checking plane #%d\n", i);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPTR (new pcl::PointCloud<pcl::PointXYZ> (segm_planes[i]));


    //STEP 3. CHECK DISTANCE. Discard if far from expected object's height



    std::stringstream ss;
    ss << "rejected_" << i << ".pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *cloudPTR, false);
    continue;

    //END OF STEP 3

    //STEP 4. Split planes on separate cluster
    std::cerr << "PointCloud representing the planar component: " << cloudPTR->width * cloudPTR->height << " data points." << std::endl;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloudPTR);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (MIN_NUM_POINTS_FOR_PLANE);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloudPTR);
    ec.extract (cluster_indices);

    // STEP 5. Check every cluster area size -> store as candidate_x if area is in thresholds
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        cloud_cluster->points.push_back (cloudPTR->points[*pit]); //*
      cloud_cluster->width = cloud_cluster->points.size ();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;

      std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
      dist_vector = calc_dist_to_plane(ground_coeffs, cloud_cluster);
      if (dist_vector[1] > 0.21 || dist_vector[1] < 0.11 || dist_vector[2] < 0.10 || dist_vector[0] > 0.25)
      {
        printf("rejected %d\n", i * 100 + l);
        std::stringstream ss;
        ss << "rejected_" << i * 100 + l << ".pcd";
        writer.write<pcl::PointXYZ> (ss.str (), *cloudPTR, false);
        continue;
      }

      float plane_area = cacl_area(cloud_cluster, j);
      if (plane_area > 0.05 )
      {
        std::stringstream ss;
        ss << "candidate_" << j << ".pcd";
        writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false);
        std::cout << ss.str () << " is stored!" << std::endl;
        j++;
      }
      l++;
    }
    //END OF STEP 4
    //END OF STEP 3
  }
  //END OF STEP 2
}



int main (int argc, char** argv)
{
  srand((unsigned int)time(NULL)); //change random seed

  pcl::PCLPointCloud2::Ptr cloud_main (new pcl::PCLPointCloud2);


  //load pcd as argument, default file otherwise
  pcl::PCDReader reader;
  if (argc > 1)
  {
    reader.read (argv[1], *cloud_main);
  }
  else
  {
    reader.read ("test_pcd.pcd", *cloud_main);
  }
  std::cerr << "PointCloud loaded has: " << cloud_main->width * cloud_main->height << " data points." << std::endl;

  clock_t tStart = clock();
  detect_planes(cloud_main);
  printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

  //test for calc area
  /*
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the cloud data
  cloud->width  = 6;
  cloud->height = 1;
  cloud->points.resize (cloud->width * cloud->height);

  // Generate the data
  for (std::size_t i = 0; i < cloud->points.size (); ++i)
  {
    //cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
    //cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
    cloud->points[i].z = 1.0;
  }

  cloud->points[0].x = 5;
  cloud->points[0].y = 5;
  cloud->points[1].x = 5;
  cloud->points[1].y = 25;

  cloud->points[2].x = 5.15;
  cloud->points[2].y = 25.2;

  cloud->points[3].x = 15;
  cloud->points[3].y = 30;

  cloud->points[4].x = 25;
  cloud->points[4].y = 25;

  cloud->points[5].x = 25;
  cloud->points[5].y = 5;

  calculateAreaPolygon(*cloud);
  */
  //pcl::PointCloud<pcl::PointXYZ>::Ptr smoothed = smooth(cloud);

  // Save output
  //pcl::io::savePCDFile ("bun0-mls.pcd", *smoothed);
}


pcl::PointCloud<pcl::PointXYZ>::Ptr smooth(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  // Create a KD-Tree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  // Output has the PointNormal type in order to store the normals calculated by MLS
  pcl::PointCloud<pcl::PointNormal> mls_points;

  // Init object (second point type is for the normals, even if unused)
  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

  mls.setComputeNormals (true);

  // Set parameters
  mls.setInputCloud (cloud);
  mls.setPolynomialOrder (2);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (0.03);

  // Reconstruct
  mls.process (mls_points);

  pcl::PointCloud<pcl::PointXYZ>::Ptr mls_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  mls_cloud->resize(mls_points.size());

  for (size_t i = 0; i < mls_points.points.size(); ++i)
  {
      mls_cloud->points[i].x = mls_points.points[i].x; //error
      mls_cloud->points[i].y = mls_points.points[i].y; //error
      mls_cloud->points[i].z = mls_points.points[i].z; //error
  }
  return mls_cloud;
}

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
  std::cerr << "Area is " << area*0.5 << std::endl;
  return area*0.5;
}

float cacl_area(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int j)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>),
                                      cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0, 1.1);
  pass.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering has: "
            << cloud_filtered->points.size () << " data points." << std::endl;
  //std::cerr << "width " << cloud_filtered->width << ", height " << cloud_filtered->height << std::endl;
  if (cloud_filtered->points.size () == 0)
  {

    return 0;
  }
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (CONV_DIST_THRESHOLD);

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

  std::cerr << "Convex hull has: " << cloud_hull->points.size ()
            << " data points." << std::endl;
  float plane_area = calculateAreaPolygon(*cloud_hull);

  pcl::PCDWriter writer;
  std::stringstream ss;
  ss << "candidate_hull_" << j << ".pcd";
  writer.write<pcl::PointXYZ> (ss.str (), *cloud_hull, false);
  std::cout << ss.str () << " is stored!" << std::endl;

  //
  //writer.write ("table_scene_mug_stereo_textured_hull.pcd", *cloud_hull, false);
  return plane_area;

}

std::vector <float> calc_dist_to_plane(std::vector <float> &ground_coeffs, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster)
{
  std::priority_queue<float> q;
  std::vector <float> to_return;
  for (int z = 0; z < POINTS_FOR_DIST_CHECK; z++)
  {
    int r_id = rand() % cloud_cluster->points.size ();
    // plane Ax + By + Cz + D = 0, point (Mx, My, Mz)
    // distance = |A*Mx + B*My + C*Mz + D| / (sqrt(A**2 + B**2 + C**2))
    float d = fabs(ground_coeffs[0]*cloud_cluster->points[r_id].x + \
      ground_coeffs[1]*cloud_cluster->points[r_id].y + ground_coeffs[2]*cloud_cluster->points[r_id].z + \
      ground_coeffs[3])/sqrt(pow(ground_coeffs[0], 2) + pow(ground_coeffs[1], 2) + pow(ground_coeffs[2], 2));
    q.push(d);
    //std::cerr << d << std::endl;
  }

  float max_d, min_d, med_d;
  int comp_for_max = q.size() - 1;
  int comp_for_med = (int)(q.size() / 2);
  int z = 0;

  while(!q.empty())
  {
    if (z == 0)
    {
      to_return.push_back(q.top());
      //printf("max %.4f\n", max_d);
    }

    else if (z == comp_for_max)
    {
      to_return.push_back(q.top());
      //printf("min %.4f\n", min_d);
    }

    else if (z == comp_for_med)
    {
      to_return.push_back(q.top());
      //printf("med %.4f\n", med_d);
    }

    q.pop();
    z++;
  }
  printf("Distance to ground - median: %.4f, max: %.4f, min: %.4f\n", to_return[1], to_return[0], to_return[2]);
  return to_return;
}
