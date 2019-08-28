#ifndef GRAPH_OPTIMIZER_HPP
#define GRAPH_OPTIMIZER_HPP

#include "configuru.hpp"

#include <map>
#include <vector>
#include <iostream>

#include <ceres/rotation.h>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <ceres/solver.h>
/*
 */
class GraphOptimizer
{
public:
    // configuration
    GraphOptimizer(configuru::Config optim_config):optim_config_(optim_config){
        dim_ = 0;
        current_num_of_vertex_ = 0;

        //options_.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options_.minimizer_progress_to_stdout = (bool) this->optim_config_["detailed_output"];
        options_.gradient_tolerance = (double) optim_config_["gradient_tolerance"];
        options_.function_tolerance = (double) optim_config_["function_tolerance"];
        options_.max_num_iterations = (int) optim_config_["max_num_iterations"];

    };

    ~GraphOptimizer(){
        if(current_num_of_vertex_ != 0) {
            delete[] vertex_values_;
        }
    }
    // data input & output (Remove is disabled due to ceres)
    // Set Num must be done before vertex adding
    void SetVertexNum(int num, int dim);
    /*
    Param type table
    0: None
    1: se3
     */
    void AddVertex(int outside_id, int local_param_type,
                   const Eigen::Ref<Eigen::MatrixXd>& initial_value);

    // Add vertex to quaternion pose
    void AddVertexQuat(int outside_id, int local_param_type,
                   const Eigen::Ref<Eigen::MatrixXd>& initial_value);

    void UpdateVertex(int outside_id, const Eigen::Ref<Eigen::MatrixXd>& update_value);

    void UpdateVertexQuat(int outside_id, const Eigen::Ref<Eigen::MatrixXd>& update_value);
    
    void AddUniEdge(int outside_id_1, int outside_id_2, 
                 ceres::CostFunction* cost_function_edge,
                 ceres::LossFunction* lost_function_edge);

    void AddDualEdge(int outside_id_1, int outside_id_2, 
                 ceres::CostFunction* cost_function_edge,
                 ceres::LossFunction* lost_function_edge);    

    // computation 
    bool Optimization();
    // Covariance Estimation
    bool CovarianceEstimation(std::vector<int>& target_ids,
                              std::map<int, Eigen::MatrixXd>& covariances);

    // log
    void Log(std::map<std::pair<int, int>, std::vector<double> >& residual_by_pair);

    void ResultOutput(Eigen::Ref<Eigen::MatrixXd> vertex_values, std::vector<int>& outside_ids);
    void ResultOutput(std::map<int, Eigen::MatrixXd>& output_values);
    
    // complementary | io tools: || template specific
    static void Quaternion2Matrix(const double* quaternion_pose, 
                           Eigen::Ref<Eigen::MatrixXd> matrix);

    static void Matrix2Quaternion(const Eigen::Ref<Eigen::MatrixXd>& matrix,
                           double* quaternion_pose); 

private:
    int dim_;
    int current_num_of_vertex_;
    int upper_bound_of_storage_;
    configuru::Config optim_config_;
    // double* 
    double* vertex_values_;

    std::map<int, int> outside2inside_;  // outside id to inside id
    std::map<int, int> inside2outside_;
    // Log structure
    std::map<std::pair<int, int>, ceres::CostFunction* > cost_function_by_pair_;

    ceres::Problem problem_;
    ceres::Solver::Options options_;
    ceres::Solver::Summary summary_;
};

#endif