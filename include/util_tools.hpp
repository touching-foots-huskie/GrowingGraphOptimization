#ifndef UTIL_TOOLS
#define UTIL_TOOLS

#include "graph_optimizer.hpp"
#include "graph_constraints.hpp"

#include <Eigen/Eigen>

void GetAveragePose(std::vector<Eigen::MatrixXd> pose_measures,
                    Eigen::Ref<Eigen::MatrixXd> result_pose) {
    // setting
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    int num_of_measures = pose_measures.size();

    // configuration
    options.linear_solver_type =  ceres::DENSE_NORMAL_CHOLESKY;
    options.gradient_tolerance = 0.001;
    options.function_tolerance = 0.001;
    options.max_num_iterations = 100;

    // problem construct
    double average_pose[7];
    GraphOptimizer::Matrix2Quaternion(pose_measures[0], average_pose);
    problem.AddParameterBlock(average_pose, 7, new ceres::ProductParameterization(
                                       new ceres::QuaternionParameterization(),
                                       new ceres::IdentityParameterization(3)));

    // Add residuals
    double Id[7] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double* > poses_for_estimation;
    for(unsigned int i = 0; i < num_of_measures; i++) {
        double* measure_pose = new double[7];
        GraphOptimizer::Matrix2Quaternion(pose_measures[i], measure_pose);
        ceres::CostFunction* measure_cost = UniPoseCost(
            measure_pose, Id, 1.0);
        problem.AddResidualBlock(measure_cost, new ceres::HuberLoss(0.1), average_pose);
    }

    ceres::Solve(options, &problem, &summary);
    GraphOptimizer::Quaternion2Matrix(average_pose, result_pose);
};

#endif
