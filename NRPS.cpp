#include "NRPS.h"
#pragma region "创建NRPS描述符"{
void NRPS_computeLocalReferenceFrame(const PointCloud<PointXYZ>::Ptr& cloud,
    pcl::PointCloud<NormalType>::Ptr cloud_normal,
    const PointXYZ& query_point,
    float rrr,
    int suporrt_z,
    int supoort_x,
    Vector3f& x_axis,
    Vector3f& y_axis,
    Vector3f& z_axis) {
    // 使用KD树进行邻域搜索
    search::KdTree<PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    std::vector<int> point_indices;
    std::vector<float> point_distances;
    // 搜索给定半径内的邻域点
    kdtree.radiusSearch(query_point, suporrt_z * rrr, point_indices, point_distances);
    size_t k = point_indices.size();

    if (k < 3) {
        std::cerr << "Not enough points in the neighborhood!" << std::endl;
        return;
    }

    // 构建协方差矩阵，使用 Matrix3f 作为 float 类型
    Matrix3f covariance_matrix;//协方差矩阵


    Vector3f centroid(query_point.x, query_point.y, query_point.z);
    // 计算加权协方差矩阵
    covariance_matrix.setZero();
    Vector3f n_mean(0, 0, 0);
    for (size_t i = 0; i < k; ++i) {
        PointXYZ& p = cloud->points[point_indices[i]];
        Vector3f diff(p.x, p.y, p.z);
        NormalType& n = cloud_normal->points[point_indices[i]];
        n_mean += Vector3f(n.normal_x, n.normal_y, n.normal_z);
        // 加权距离
        float weight = (rrr * suporrt_z - sqrt(point_distances[i])) * (rrr * suporrt_z - sqrt(point_distances[i]));

        diff -= centroid;

        covariance_matrix += weight * (diff * diff.transpose());
    }

    // 进行特征值分解（EVD）
    SelfAdjointEigenSolver<Matrix3f> solver(covariance_matrix);

    if (!solver.info())
    {

        z_axis = solver.eigenvectors().col(0).cast<float>();// Smallest eigenvalue's eigenvector

        if (n_mean.norm() > 0)
        {
            n_mean = n_mean.normalized();
        }
        else
        {
            // 如果 n_mean 为零向量，则无法确定正确的 z 轴方向
            PCL_ERROR("[fmj_computeLocalReferenceFrame] Error! Mean normal is zero vector.\n");
            x_axis.setConstant(std::numeric_limits<float>::quiet_NaN());
            y_axis.setConstant(std::numeric_limits<float>::quiet_NaN());
            z_axis.setConstant(std::numeric_limits<float>::quiet_NaN());
            return;
        }
        if (z_axis.dot(n_mean) < 0)
        {
            z_axis = -z_axis;
        }
    }
    else
    {
        PCL_ERROR("[SHOT_computeLocalReferenceFrame] Error! Eigen decomposition failed.\n");
        x_axis.setConstant(std::numeric_limits<float>::quiet_NaN());
        y_axis.setConstant(std::numeric_limits<float>::quiet_NaN());
        z_axis.setConstant(std::numeric_limits<float>::quiet_NaN());
    }
    //求x轴
    kdtree.radiusSearch(query_point, supoort_x * rrr, point_indices, point_distances);
    k = point_indices.size();
    int idx_x = 0;
    float n_min = 1;
    for (int i = 1; i < k; i++)
    {
        NormalType& n = cloud_normal->points[point_indices[i]];
        float n_temp = z_axis.dot(Vector3f(n.normal_x, n.normal_y, n.normal_z));
        if (n_temp < n_min)
        {
            n_min = n_temp;
            idx_x = i;
        }
    }
    x_axis = cloud->points[point_indices[idx_x]].getVector3fMap() - query_point.getVector3fMap();
    //减去在z轴上的投影
    x_axis = x_axis - x_axis.dot(z_axis) * z_axis;
    x_axis.normalize();
    y_axis = z_axis.cross(x_axis);
}
void local_NRPS(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, vector<int>& pointIdx, vector<float>& pointDst, float sup_radius, int bin_num, LRF_NRPS& p_lrf, Eigen::Matrix4f transform, vector<float>& histogram)
{
    //cloud里面有坐标和法向量
    //根据法向量和索引
    {
        // 1. 变换点云到局部坐标系
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_neighbor(new pcl::PointCloud<pcl::PointNormal>);
        pcl::transformPointCloudWithNormals(*cloud, pointIdx, *cloud_neighbor, transform);

        // 2. 初始化直方图
        const int total_bins = 9 * bin_num;
        histogram.resize(total_bins, 0.0f);

        // 3. 计算每个点的贡献
        for (int i = 0; i < pointIdx.size(); i++)
        {

            // 使用pointDst计算距离
            float rou = sqrt(pointDst[i]);
            if (rou > sup_radius) continue;  // 跳过超出支撑半径的点

            // 计算点在xy平面上的极坐标
            float x = cloud_neighbor->points[i].x;
            float y = cloud_neighbor->points[i].y;
            float theta = std::atan2(y, x);  // 与x轴的夹角 (-π到π)

            // 将theta转换到0到2π范围
            if (theta < 0) theta += 2 * M_PI;

            // 4. 计算空间区域的软分配
            vector<pair<int, float>> spatial_contributions;//存储空间分配的权重
            if (rou < sup_radius / 2.0f) {
                // 如果点接近内外圆的边界，同时贡献给两个区域
                float boundary_width = sup_radius * 0.05f;  // 边界宽度，可调整
                if (rou > (sup_radius / 2.0f - boundary_width)) {
                    float inner_weight = (sup_radius / 2.0f - rou) / boundary_width;//内点权重
                    float outer_weight = 1.0f - inner_weight;//外点权重

                    // 计算外圈区域索引
                    float angle_step = 2 * M_PI / 8;
                    int outer_bin = 1 + static_cast<int>(theta / angle_step);
                    if (outer_bin == 9) outer_bin = 8;

                    spatial_contributions.push_back({ 0, inner_weight });
                    spatial_contributions.push_back({ outer_bin, outer_weight });
                }
                else {
                    spatial_contributions.push_back({ 0, 1.0f });
                }
            }
            else {
                // 外环8个区域的软分配
                float angle_step = 2 * M_PI / 8;
                int main_bin = 1 + static_cast<int>(theta / angle_step);
                if (main_bin == 9) main_bin = 8;

                // 计算角度在bin边界的位置
                float bin_center = (main_bin - 1) * angle_step + angle_step / 2;
                float angle_diff = fabs(theta - bin_center);

                if (angle_diff < angle_step * 0.25f) {
                    // 如果点距离bin中心较近，只贡献给这个bin
                    spatial_contributions.push_back({ main_bin, 1.0f });
                }
                else {
                    // 否则贡献给相邻的两个bin
                    float main_weight = (angle_step * 0.5f - angle_diff) / (angle_step * 0.25f);
                    float neighbor_weight = 1.0f - main_weight;

                    spatial_contributions.push_back({ main_bin, main_weight });

                    // 确定相邻bin
                    int neighbor_bin;
                    if (theta < bin_center) {
                        neighbor_bin = (main_bin == 1) ? 8 : (main_bin - 1);
                    }
                    else {
                        neighbor_bin = (main_bin == 8) ? 1 : (main_bin + 1);
                    }
                    spatial_contributions.push_back({ neighbor_bin, neighbor_weight });
                }
            }

            // 5. 计算法向量在xy平面的投影方向
            float nx = cloud_neighbor->points[i].normal_x;
            float ny = cloud_neighbor->points[i].normal_y;
            float normal_theta = std::atan2(ny, nx);
            if (normal_theta < 0) normal_theta += 2 * M_PI;

            // 6. 计算方向空间的软分配
            vector<pair<int, float>> direction_contributions;
            float angle_step = 2 * M_PI / bin_num;
            int main_direction = static_cast<int>(normal_theta / angle_step);
            if (main_direction == bin_num) main_direction = bin_num - 1;

            // 计算方向在bin边界的位置
            float bin_center = main_direction * angle_step + angle_step / 2;
            float angle_diff = fabs(normal_theta - bin_center);

            if (angle_diff < angle_step * 0.25f) {
                direction_contributions.push_back({ main_direction, 1.0f });
            }
            else {
                float main_weight = (angle_step * 0.5f - angle_diff) / (angle_step * 0.25f);
                float neighbor_weight = 1.0f - main_weight;

                direction_contributions.push_back({ main_direction, main_weight });

                // 确定相邻bin
                int neighbor_direction;
                if (normal_theta < bin_center) {
                    neighbor_direction = (main_direction == 0) ? (bin_num - 1) : (main_direction - 1);
                }
                else {
                    neighbor_direction = (main_direction == bin_num - 1) ? 0 : (main_direction + 1);
                }
                direction_contributions.push_back({ neighbor_direction, neighbor_weight });
            }

            // 7. 计算距离权重
            float distance_weight = sqrt(nx * nx + ny * ny);
            // 8. 更新直方图，考虑所有贡献
            for (const auto& spatial : spatial_contributions) {
                for (const auto& direction : direction_contributions) {
                    int bin_index = spatial.first * bin_num + direction.first;
                    float weight = distance_weight * spatial.second * direction.second;
                    histogram[bin_index] += weight;
                }
            }
        }

        // 9. 归一化直方图
        float sum = 0.0f;
        for (float& bin : histogram) {
            sum += bin;
        }
        if (sum > 0) {
            for (float& bin : histogram) {
                bin /= sum;
            }
        }
    }
}
void compute_rotated_NRPS(pcl::PointCloud<pcl::PointNormal>::Ptr cloud,
    vector<int>& pointIdx,
    vector<float>& pointDst,
    float sup_radius,
    int bin_num,
    LRF_NRPS& p_lrf,
    pcl::PointNormal& query_point,
    vector<float>& combined_feature,
    float rotated_angle,
    int rotated_num)
{
    // 1. 构建初始的局部坐标系变换矩阵
    Eigen::Matrix3f R;
    R.col(0) = p_lrf.x_axis;
    R.col(1) = p_lrf.y_axis;
    R.col(2) = p_lrf.z_axis;

    // 2. 定义旋转角度数组（度数）
    //vector<float> angles = { 0, 30, 60, 90, 120, 150, 180 };
    //利用旋转角度rotated_angle 和旋转次数rotated_num构建vector<float> angles
    vector<float> angles;
    for (int i = 0; i < rotated_num; i++)
    {
        angles.push_back(rotated_angle * i);
    }

    // 3. 对每个角度计算特征并串联
    for (float angle : angles)
    {
        // 将角度转换为弧度
        float rad = angle * M_PI / 180.0f;

        // 构建三轴同时旋转的旋转矩阵
        // 使用Rodriguez旋转公式构建旋转矩阵
        Eigen::Matrix3f Rx, Ry, Rz;

        // 绕X轴旋转
        Rx << 1, 0, 0,
            0, cos(rad), -sin(rad),
            0, sin(rad), cos(rad);

        // 绕Y轴旋转
        Ry << cos(rad), 0, sin(rad),
            0, 1, 0,
            -sin(rad), 0, cos(rad);

        // 绕Z轴旋转
        Rz << cos(rad), -sin(rad), 0,
            sin(rad), cos(rad), 0,
            0, 0, 1;

        // 组合旋转：先绕X轴，再绕Y轴，最后绕Z轴
        Eigen::Matrix3f R_combined = Rz * Ry * Rx * R.transpose();

        // 构建4x4变换矩阵
        Eigen::Matrix4f transform_LRF = Eigen::Matrix4f::Identity();
        // 设置旋转部分
        transform_LRF.block<3, 3>(0, 0) = R_combined;
        // 设置平移部分：p' = R^T * (p - t)  =>  t = -R * p
        transform_LRF.block<3, 1>(0, 3) = -R_combined * Eigen::Vector3f(query_point.x, query_point.y, query_point.z);

        // 计算当前角度的特征
        vector<float> current_feature;
        local_FMJ(cloud, pointIdx, pointDst, sup_radius, bin_num, p_lrf, transform_LRF, current_feature);

        // 将当前特征添加到组合特征末尾
        combined_feature.insert(combined_feature.end(), current_feature.begin(), current_feature.end());
    }

}
void NRPS_compute(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud1, vector<int>& indices, pcl::PointCloud<NormalType>::Ptr cloud_normal, float sup_radius, int bin_num, vector<vector<float>>& Histograms, float rrr, float rotated_angle, int rotated_num,
    int suporrt_z, int supoort_x)
{
    //建立点云的索引
    pcl::KdTreeFLANN<pcl::PointNormal> kdtree_cloud;
    kdtree_cloud.setInputCloud(cloud);
    vector<int> pointIdx;//存储描述子支撑域内的点云id
    vector<float> pointDst;//存储描述子支撑域内的点云距离
    pcl::PointNormal query_point;
    LRF_NRPS p_lrf;
    for (int i = 0; i < indices.size(); i++)
    {
        query_point = cloud->points[indices[i]];
        p_lrf.pointID = indices[i];
        fmj_computeLocalReferenceFrame(cloud1, cloud_normal, query_point, rrr, suporrt_z, supoort_x, p_lrf.x_axis, p_lrf.y_axis, p_lrf.z_axis);
        if (!pcl_isfinite(p_lrf.x_axis[0]) || !pcl_isfinite(p_lrf.y_axis[0]) || !pcl_isfinite(p_lrf.z_axis[0]))
        {
            vector<float> combined_feature;
            combined_feature.resize(9 * bin_num * rotated_num, 0.0f); // 7个角度
            Histograms.push_back(combined_feature);
            continue;

        }
        
        if (kdtree_cloud.radiusSearch(query_point, sup_radius, pointIdx, pointDst) > 10)
        {
            vector<float> combined_feature;
            compute_rotated_FMJ(cloud, pointIdx, pointDst, sup_radius, bin_num,
                p_lrf, query_point, combined_feature, rotated_angle, rotated_num);
            Histograms.push_back(combined_feature);
        }
        else {
            vector<float> combined_feature;
            combined_feature.resize(9 * bin_num * rotated_num, 0.0f); // 7个角度
            Histograms.push_back(combined_feature);
        }
    }
}
#pragma endregion "创建NRPS描述符"}





