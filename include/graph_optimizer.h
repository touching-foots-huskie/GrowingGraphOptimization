#ifndef GRAPH_OPTIMIZER
#define GRAPH_OPTIMIZER

#include "local_parameterization_se3.hpp"
#include "configuru.hpp"

#include <map>
#include <vector>
#include <iostream>

#include <sophus/se3.hpp>
#include <Eigen/Eigen>
#include <ceres/ceres.h>

class GraphOptimizer
{
public:
    // configuration
    GraphOptimizer(configuru::Config optim_config):optim_config_(optim_config){
        dim_ = 0;
        total_num_of_vertex_ = 0;
        current_num_of_vertex_ = 0;

        options_.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options_.minimizer_progress_to_stdout = (bool) this->optim_config_["detailed_output"];
        /*
        options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        */
        options_.gradient_tolerance = 1e-5;
        options_.function_tolerance = 1e-5;
        options_.max_num_iterations = 1000;

    };

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

    void UpdateVertex(int outside_id, const Eigen::Ref<Eigen::MatrixXd>& update_value);

    void AddUniEdge(int outside_id_1, int outside_id_2, 
                 ceres::CostFunction* cost_function_edge,
                 ceres::LossFunction* lost_function_edge);

    void AddDualEdge(int outside_id_1, int outside_id_2, 
                 ceres::CostFunction* cost_function_edge,
                 ceres::LossFunction* lost_function_edge);
    

    // computation 
    bool Optimization();

    // log
    void Log();

    void ResultOutput(Eigen::Ref<Eigen::MatrixXd> vertex_values, std::vector<int>& outside_ids);

private:
    int dim_;
    int total_num_of_vertex_;
    int current_num_of_vertex_;
    configuru::Config optim_config_;
    Eigen::MatrixXd vertex_values_;  // Dim:[D, N]

    std::map<int, int> outside2inside_;  // outside id to inside id
    std::map<int, int> inside2outside_;

    ceres::Problem problem_;
    ceres::Solver::Options options_;
    ceres::Solver::Summary summary_;

};

#endif