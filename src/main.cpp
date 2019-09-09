#include "graph_optimizer.hpp"
#include "graph_constraints.hpp"
#include "util_tools.hpp"
// configuration process
#define CONFIGURU_IMPLEMENTATION 1
#include "configuru.hpp"

#include <cmath>
#include <iostream>

#include <Eigen/Eigen>

int main () {
    std::string optim_config_path = "/home/harvey/CppProjects/GrowingGraphOptimization/config/optim_config.json";
    configuru::Config optim_config = configuru::parse_file(optim_config_path, configuru::JSON);
    GraphOptimizer graph_optimizer(optim_config);

    // test 1: Pose Cost Functor
    Eigen::MatrixXd object_pose_1(4, 4);
    object_pose_1 << 1, 0, 0, 1,
                     0, 1, 0, 2, 
                     0, 0, 1, 3,
                     0, 0, 0, 1;
    
    Eigen::MatrixXd object_pose_2(4, 4);
    object_pose_2 << 0.5, std::sqrt(3) / 2.0, 0, 0, 
                     - std::sqrt(3), 0.5, 0, 1, 
                     0, 0, 1, 0, 
                     0, 0, 0, 1;
    
    Eigen::MatrixXd relative_pose_1to2(4, 4);
    relative_pose_1to2 << std::sqrt(2) / 2.0, std::sqrt(2) / 2.0, 0, -0.8, 
                          - std::sqrt(2) / 2.0, std::sqrt(2) / 2.0, 0, -1.2,
                          0, 0, 1, -3.2,
                          0, 0, 0, 1;

    graph_optimizer.SetVertexNum(2, 7);
    graph_optimizer.AddVertexQuat(1, 1, object_pose_1);
    graph_optimizer.AddVertexQuat(2, 1, object_pose_2);

    // relative pose
    // transform into double*
    double quat_object_pose_1[7];
    GraphOptimizer::Matrix2Quaternion(object_pose_1, quat_object_pose_1);
    double relative_pose[7];

    GraphOptimizer::Matrix2Quaternion(relative_pose_1to2, relative_pose);
    // check programming
    for(int i = 0; i < 7; i++) {
        std::cout << relative_pose[i] << " , ";
    }
    std::cout << std::endl;

    ceres::CostFunction* dual_pose_cost = UniSymmetricCost(quat_object_pose_1, relative_pose, 1.0, 1.0);
    graph_optimizer.AddUniEdge(2, 1, dual_pose_cost, new ceres::TrivialLoss());
    graph_optimizer.Optimization();

    // Output Result
    std::map<int, Eigen::MatrixXd> output_values;
    graph_optimizer.ResultOutput(output_values);

    // then we can check the optimized result
    Eigen::MatrixXd optimized_relative_pose(4, 4);
    optimized_relative_pose =  output_values[1].inverse() * output_values[2];
    std::cout << "------ Result in 1st stage ------" << std::endl;
    std::cout << optimized_relative_pose << std::endl;
    // test 2 : Plane2Plane Constraints Test & test 3 : Plane2Surf Constraints Test
    double normal_1[3] = {0, 0, 1};
    double normal_2[3] = {0, 0, 1};
    double center_1[3] = {0, 0, 0};
    double center_2[3] = {0, 0, 0};
    double weight_1 = 100.0;  // Give it a larger result
    double weight_2 = 100.0;
    /*
    ceres::CostFunction* plane2plane_cost = DualPlane2PlaneCost(normal_1, center_1,
                                                                normal_2, center_2,
                                                                weight_1, weight_2);

    graph_optimizer.AddDualEdge(1, 2, plane2plane_cost, new ceres::TrivialLoss());
     */
    ceres::CostFunction* plane2surf_cost = DualPlane2SurfCost(normal_1, center_1,
                                                              normal_2, center_2,
                                                              1.0,
                                                              weight_1, weight_2);

    graph_optimizer.AddDualEdge(1, 2, plane2surf_cost, new ceres::TrivialLoss());
    graph_optimizer.Optimization();

    // Output Result
    output_values.clear();
    graph_optimizer.ResultOutput(output_values);

    // then we can check the optimized result
    optimized_relative_pose =  output_values[1].inverse() * output_values[2];
    std::cout << "------ Result in 2nd stage ------" << std::endl;
    std::cout << optimized_relative_pose << std::endl;                                                            

    //
    // Test Estimation
    std::map<int, Eigen::MatrixXd> out_covariance;
    std::vector<int> target_ids = {1};
    if(graph_optimizer.CovarianceEstimation(target_ids, out_covariance)) {
        std::cout << "---- Covariance -------" << std::endl;
        std::cout << out_covariance[2] << std::endl;
    }
    
    // log check
    /*
    std::map<std::pair<int, int>, std::vector<double> > residual_by_pair;
    graph_optimizer.Log(residual_by_pair);
    for(auto it = residual_by_pair.begin(); it != residual_by_pair.end(); ++it) {
        std::cout << it->first.first << " , " << it->first.second << std::endl;
        for(auto residual : it->second) {
            std::cout << residual << ", ";
        }
        std::cout << std::endl;
    }
     */
    // test measure average
    std::vector<Eigen::MatrixXd> object_poses;
    Eigen::MatrixXd result_pose(4, 4);
    object_poses.emplace_back(object_pose_1);
    object_poses.emplace_back(object_pose_2);
    GetAveragePose(object_poses, result_pose);
    std::cout << "------ Result in 3rd stage ------" << std::endl;
    std::cout << result_pose << std::endl;
    
    return 0;
}