#pragma once

// PCL�����������ƴ���
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// PCL I/O����
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// PCL������ȡ
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/features/shot.h>
#include <pcl/features/board.h>
#include <pcl/features/rops_estimation.h>

// PCL�ؼ�����
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/keypoints/harris_3D.h>

// PCL�����ؽ�
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>

// PCLע����任
#include <pcl/registration/transforms.h>
#include <pcl/common/transforms.h>

// PCL KDTree
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

// PCL��������
#include <pcl/correspondence.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/distances.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>

// PCL���ӻ�
#include <pcl/visualization/pcl_visualizer.h>

// C++��׼��
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

// Eigen��
#include <Eigen/Dense>

// PCL����̨����
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#define pcl_isfinite(x) std::isfinite(x)
#define PI std::numbers::pi
#define Pi 3.1415926
#define NULL_POINTID -1
#define TOLDI_NULL_PIXEL 100
typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
using namespace std;
using namespace pcl;
using namespace Eigen;
#pragma region "����NRPS������"{
typedef struct {
    int pointID;
    Vector3f x_axis;
    Vector3f y_axis;
    Vector3f z_axis;
}LRF_NRPS;
void NRPS_computeLocalReferenceFrame(const PointCloud<PointXYZ>::Ptr& cloud,
    pcl::PointCloud<NormalType>::Ptr cloud_normal,
    const PointXYZ& query_point,
    float rrr,
    int suporrt_z,
    int supoort_x,
    Vector3f& x_axis,
    Vector3f& y_axis,
    Vector3f& z_axis);
void local_NRPS(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, vector<int>& pointIdx, vector<float>& pointDst, float sup_radius, int bin_num, LRF_NRPS& p_lrf, Eigen::Matrix4f transform, vector<float>& histogram);
void compute_rotated_NRPS(pcl::PointCloud<pcl::PointNormal>::Ptr cloud,
    vector<int>& pointIdx,
    vector<float>& pointDst,
    float sup_radius,
    int bin_num,
    LRF_NRPS& p_lrf,
    pcl::PointNormal& query_point,
    vector<float>& combined_feature,
    float rotated_angle,
    int rotated_num);
void NRPS_compute(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud1, vector<int>& indices, pcl::PointCloud<NormalType>::Ptr cloud_normal, float sup_radius, int bin_num, vector<vector<float>>& Histograms, float rrr, float rotated_angle, int rotated_num,
    int suporrt_z, int supoort_x);
#pragma endregion "����NRPS������"}
