#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/filter.h>

#include <vector>
#include <string>
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

#define SEGM_DIST_THRESHOLD 0.05// 0.1
#define CONV_DIST_THRESHOLD 0.01//0.01
#define MIN_NUM_POINTS_FOR_PLANE 100
#define POINTS_FOR_DIST_CHECK 31 // TO BE ODD!

//#define DEBUG
//supportive funstions - implementation is below main()
pcl::PointCloud<pcl::PointXYZ>::Ptr smooth(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
std::vector <float> calculateAreaPolygon(const pcl::PointCloud<pcl::PointXYZ> &polygon);
std::vector <float> cacl_area(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int j);
std::vector <float> calc_dist_to_plane(std::vector <float> &ground_coeffs, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster);

//main algorithm
std::string detect_planes(pcl::PCLPointCloud2::Ptr cloud_blob, int frame_id)
{
  std::string toRet;
  //Step 1. Segment all planes
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCLPointCloud2::Ptr cloud_filtered_blob (new pcl::PCLPointCloud2);

  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud_blob);
  sor.setLeafSize (0.02f, 0.02f, 0.02f);
  sor.filter (*cloud_filtered_blob);

  // Convert to the templated PointCloud
  pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);
  if (cloud_filtered->width * cloud_filtered->height == 0)
  {
    printf("NO POINTS ON PLANE! RETURN\n");
    return toRet;
  }


  #ifdef DEBUG
  std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;

  // Create writer

  pcl::PCDWriter writer;
  std::stringstream ss_init;
  ss_init << "plane_" << frame_id << ".pcd";
  writer.write<pcl::PointXYZ> (ss_init.str (), *cloud_filtered, false);
  #endif

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
  int i = 0, nr_points = (int) cloud_filtered->points.size ();
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
    if (segm_planes.size() == 0)
    {
      ground_coeffs.push_back(coefficients->values[0]);
      ground_coeffs.push_back(coefficients->values[1]);
      ground_coeffs.push_back(coefficients->values[2]);
      ground_coeffs.push_back(coefficients->values[3]);
    }



    segm_planes.push_back(*cloud_p);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);
    i++;
  }

  printf("Found %d planes\n", (int)segm_planes.size());
  if ((int)segm_planes.size() == 1)
  {
    printf("JUST GROUND PLANE! RETURN\n");
    return toRet;
  }

  // END OF STEP 1



  //STEP 2. CHECK PLANES
  int j = 0, l = 0;
  for (i = 1; i< segm_planes.size() ; i++)
  {
    printf("Checking plane #%d\n", i);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPTR (new pcl::PointCloud<pcl::PointXYZ> (segm_planes[i]));
    std::vector <float> dist_vector;

    #ifdef DEBUG
    std::stringstream ss_tmp;
    ss_tmp << "plane_" << i << ".pcd";
    writer.write<pcl::PointXYZ> (ss_tmp.str (), *cloudPTR, false);
    #endif

    //STEP 3. CHECK DISTANCE. Discard if far from expected object's height

    dist_vector = calc_dist_to_plane(ground_coeffs, cloudPTR);
    if (dist_vector[1] > 0.23 || dist_vector[1] < 0.08)
    {
      printf("rejected %d\n", i);

      #ifdef DEBUG
      std::stringstream ss;
      ss << "rejected_" << i << ".pcd";
      writer.write<pcl::PointXYZ> (ss.str (), *cloudPTR, false);
      #endif
      continue;
    }
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

      l++;

      std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
      dist_vector = calc_dist_to_plane(ground_coeffs, cloud_cluster);
      if ((dist_vector[0] > 0.25 || dist_vector[2] < 0.05) || dist_vector[1] < 0.09)
      {
        printf("rejected %d\n", i * 100 + l);
        #ifdef DEBUG
        std::stringstream ss;
        ss << "rejected_" << i * 100 + l << ".pcd";
        writer.write<pcl::PointXYZ> (ss.str (), *cloudPTR, false);
        #endif
        continue;
      }
      std::vector <float> area_center = cacl_area(cloud_cluster, j);

      if (area_center[0] > 0.01 ) //0.05
      {
        #ifdef DEBUG
        std::stringstream ss;
        ss << "candidate_" << j << ".pcd";

        writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false);
        std::cout << ss.str () << " is stored!" << std::endl;
        #endif


        toRet += std::to_string(area_center[0]);
        toRet += "#";
        toRet += std::to_string(area_center[1]);
        toRet += "#";
        toRet += std::to_string(area_center[2]);
        toRet += "#";
        toRet += std::to_string(area_center[3]);
        toRet += "x";

        j++;
      }

    }
    //END OF STEP 4
    //END OF STEP 3
  }
  //END OF STEP 2
  return toRet;
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

std::vector <float> calculateAreaPolygon(const pcl::PointCloud<pcl::PointXYZ> &polygon )
{
  std::vector <float> result;
  float area=0.0;
  int num_points = polygon.size();
  int j = 0;
  float x_c = 0, y_c = 0, z_c = 0;
  Eigen::Vector3f va,vb,res;
  res(0) = res(1) = res(2) = 0.0f;
  for (int i = 0; i < num_points; ++i)
  {
      x_c += polygon.points[i].x;
      y_c += polygon.points[i].y;
      z_c += polygon.points[i].z;
      j = (i + 1) % num_points;
      va = polygon[i].getVector3fMap();
      vb = polygon[j].getVector3fMap();
      res += va.cross(vb);
      //std::cerr << va << std::endl;
  }
  area=res.norm();
  x_c /= num_points;
  y_c /= num_points;
  z_c /= num_points;
  result.push_back(area*0.5);
  result.push_back(x_c);
  result.push_back(y_c);
  result.push_back(z_c);
  std::cerr << "Area is " << result[0] << std::endl;
  return result;
}

std::vector <float> cacl_area(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int j)
{
  std::vector <float> result(4,0);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>),
                                      cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (-5, 5);
  pass.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering has: "
            << cloud_filtered->points.size () << " data points." << std::endl;
  //std::cerr << "width " << cloud_filtered->width << ", height " << cloud_filtered->height << std::endl;
  if (cloud_filtered->points.size () == 0)
  {

    return result;
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


  //chull.setComputeAreaVolume(true);

  std::cerr << "Convex hull has: " << cloud_hull->points.size ()
            << " data points." << std::endl;
  result = calculateAreaPolygon(*cloud_hull);

  #ifdef DEBUG
  pcl::PCDWriter writer;
  std::stringstream ss;
  ss << "candidate_hull_" << j << ".pcd";
  writer.write<pcl::PointXYZ> (ss.str (), *cloud_hull, false);
  std::cout << ss.str () << " is stored!" << std::endl;
  #endif

  //
  //writer.write ("table_scene_mug_stereo_textured_hull.pcd", *cloud_hull, false);
  return result;

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
