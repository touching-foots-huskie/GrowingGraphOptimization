#include "graph_optimizer.hpp"

// Set Num must be done before vertex adding
void GraphOptimizer::SetVertexNum(int num, int dim){
    assert(current_num_of_vertex_ == 0);
    dim_ = dim;
    upper_bound_of_storage_ = 2 * num;
    vertex_values_ = new double[upper_bound_of_storage_ * dim];
};

// You'd better to set Vertex Num first
// Initial value Dim:[D, 1]
void GraphOptimizer::AddVertex(int outside_id, int local_param_type,
                               const Eigen::Ref<Eigen::MatrixXd>& initial_value){
    
    outside2inside_[outside_id] = current_num_of_vertex_;
    inside2outside_[current_num_of_vertex_] = outside_id;

    if(dim_ == 0){
        // if didn't update dim
        dim_ = initial_value.rows();
    }
    else{
        assert(initial_value.rows() == dim_);
    }
    
    if(current_num_of_vertex_ < upper_bound_of_storage_){
        for(int i = 0; i < dim_; i++) {
            vertex_values_[current_num_of_vertex_ * dim_ + i] = initial_value(i, 0);
        }
    }
    else {
        // copy previous data & expand upper bound
        upper_bound_of_storage_ *= 2;
        double* new_vertex_values_ = new double[upper_bound_of_storage_ * dim_];
        // TODO: If I can directly move the meomery ?
        for(int i = 0; i < current_num_of_vertex_ * dim_; i++) {
            new_vertex_values_[i] = vertex_values_[i];
        }
        delete[] vertex_values_;
        vertex_values_ = new_vertex_values_;
        // expand total vertex value
        for(int i = 0; i < dim_; i++) {
            vertex_values_[current_num_of_vertex_ * dim_ + i] = initial_value(i, 0);
        }
    }
    
    current_num_of_vertex_ ++;

    // add paramblock
    switch (local_param_type)
    {
    case 0:
        break;
    case 1:
        assert(dim_ == 7);
        // TODO: If we really need this quaternion?

        problem_.AddParameterBlock(&vertex_values_[(current_num_of_vertex_ - 1)* dim_], 
                                   dim_, 
                                   new ceres::ProductParameterization(
                                       new ceres::QuaternionParameterization(),
                                       new ceres::IdentityParameterization(3)));
        break;
    default:
        std::cout << "No such local param" << std::endl;
        break;
    }
};

void GraphOptimizer::AddVertexQuat(int outside_id, int local_param_type,
                   const Eigen::Ref<Eigen::MatrixXd>& initial_value) {
    outside2inside_[outside_id] = current_num_of_vertex_;
    inside2outside_[current_num_of_vertex_] = outside_id;
    
    if(current_num_of_vertex_ < upper_bound_of_storage_){
        Matrix2Quaternion(initial_value, &vertex_values_[current_num_of_vertex_ * dim_]);
    }
    else {
        // copy previous data & expand upper bound
        upper_bound_of_storage_ *= 2;
        double* new_vertex_values_ = new double[upper_bound_of_storage_ * dim_];
        // TODO: If I can directly move the meomery ?
        for(int i = 0; i < current_num_of_vertex_ * dim_; i++) {
            new_vertex_values_[i] = vertex_values_[i];
        }
        delete[] vertex_values_;
        vertex_values_ = new_vertex_values_;
        // expand total vertex value
        Matrix2Quaternion(initial_value, &vertex_values_[current_num_of_vertex_ * dim_]);
    }
    
    current_num_of_vertex_ ++;

    // add paramblock
    switch (local_param_type)
    {
    case 0:
        break;
    case 1:
        assert(dim_ == 7);
        // TODO: If we really need this quaternion?

        problem_.AddParameterBlock(&vertex_values_[(current_num_of_vertex_ - 1)* dim_], 
                                   dim_, 
                                   new ceres::ProductParameterization(
                                       new ceres::QuaternionParameterization(),
                                       new ceres::IdentityParameterization(3)));
        break;
    default:
        std::cout << "No such local param" << std::endl;
        break;
    }
};

void GraphOptimizer::UpdateVertex(int outside_id, const Eigen::Ref<Eigen::MatrixXd>& update_value){
    int inside_id = outside2inside_.at(outside_id);
    for(int i = 0; i < dim_; i++) {
        vertex_values_[inside_id * dim_ + i] = update_value(i, 0);
    }

};

void GraphOptimizer::UpdateVertexQuat(int outside_id, const Eigen::Ref<Eigen::MatrixXd>& update_value){
    int inside_id = outside2inside_.at(outside_id);
    Matrix2Quaternion(update_value, &vertex_values_[inside_id * dim_]);
};

// Uni Edge only update data 1
void GraphOptimizer::AddUniEdge(int outside_id_1, int outside_id_2, 
                             ceres::CostFunction* cost_function_edge,
                             ceres::LossFunction* lost_function_edge){

    int inside_id_1 = outside2inside_.at(outside_id_1);
    cost_function_by_pair_[std::pair<int, int>(outside_id_1, outside_id_2)]
        =   cost_function_edge;

    problem_.AddResidualBlock(cost_function_edge, lost_function_edge, 
                              &vertex_values_[inside_id_1 * dim_]);
};

// Dual Edge update both 1 & 2
void GraphOptimizer::AddDualEdge(int outside_id_1, int outside_id_2, 
                             ceres::CostFunction* cost_function_edge,
                             ceres::LossFunction* lost_function_edge){

    int inside_id_1 = outside2inside_.at(outside_id_1);
    int inside_id_2 = outside2inside_.at(outside_id_2);
    cost_function_by_pair_[std::pair<int, int>(outside_id_1, outside_id_2)]
        =   cost_function_edge;

    problem_.AddResidualBlock(cost_function_edge, lost_function_edge, 
                              &vertex_values_[inside_id_1 * dim_],
                              &vertex_values_[inside_id_2 * dim_]);
};

bool GraphOptimizer::Optimization(){
    ceres::Solve(options_, &problem_, &summary_);
    if((bool) optim_config_["report_output"]){
        std::cout << summary_.BriefReport() << "\n";
    }
    
    // TODO: decide if optimization is finished
    return true;
};

bool GraphOptimizer::CovarianceEstimation(std::vector<int>& target_ids,
                                          std::map<int, Eigen::MatrixXd>& covariances) {
    // config
    ceres::Covariance::Options cov_option;
    cov_option.null_space_rank = -1;
    cov_option.algorithm_type = ceres::DENSE_SVD;
    ceres::Covariance covariance(cov_option);
    std::vector<std::pair<const double*, const double*> > covariance_blocks;

    // add covariance:
    for(unsigned int i = 0; i < target_ids.size(); i++) {
        int index = outside2inside_.at(target_ids[i]);
        covariance_blocks.emplace_back(
            std::make_pair(
                &vertex_values_[dim_ * index],
                &vertex_values_[dim_ * index]
            )
        );
    }

    // Compute
    if(covariance.Compute(covariance_blocks, &problem_)) {
        // Log
        for(unsigned int i = 0; i < target_ids.size(); i++) {
            int index = outside2inside_.at(target_ids[i]);
            double covaraince_xx[dim_ * dim_];
            covariance.GetCovarianceBlock(
                &vertex_values_[dim_ * index],
                &vertex_values_[dim_ * index],
                covaraince_xx
            );
            
            // log
            Eigen::MatrixXd eigen_covariance(dim_, dim_);
            for(int j = 0; j < dim_; j++) {
                for(int k = 0; k < dim_; k++) {
                    eigen_covariance(j, k) = 
                    covaraince_xx[dim_ * j + k]; 
                }
            }
            covariances[target_ids[i]] = eigen_covariance;
        }
        return true;
    }
    else {
        std::cout << "Covariance Ill" << std::endl;
        return false;
    }
};

/*
Log is giving a detailed output of costs
 */
void GraphOptimizer::Log(std::map<std::pair<int, int>, std::vector<double> >& residual_by_pair) {
    for(auto it = cost_function_by_pair_.begin(); it != cost_function_by_pair_.end(); ++it) {
        if(it->first.second == -1) {
            // an uniedge
            int inside_id_1 = outside2inside_.at(it->first.first);
            // get params
            double* params[1];
            params[0] = &vertex_values_[inside_id_1 * dim_];
            // get residual
            int num_of_residual = it->second->num_residuals();
            double* residual =  new double[num_of_residual];
            it->second->Evaluate(params, residual, nullptr);
            std::vector<double> result_by_cost_function;
            for(int i = 0; i < num_of_residual; i++) {
                result_by_cost_function.emplace_back(residual[i]);
            }
            delete[] residual;  // clear meomery
            residual_by_pair[it->first] = result_by_cost_function;
        }
        else {
            // a dual edge
            int inside_id_1 = outside2inside_.at(it->first.first);
            int inside_id_2 = outside2inside_.at(it->first.second);
            double* params[2];
            params[0] = &vertex_values_[inside_id_1 * dim_];
            params[1] = &vertex_values_[inside_id_2 * dim_];
            // get residual
            int num_of_residual = it->second->num_residuals();
            double* residual =  new double[num_of_residual];
            it->second->Evaluate(params, residual, nullptr);
            std::vector<double> result_by_cost_function;
            for(int i = 0; i < num_of_residual; i++) {
                result_by_cost_function.emplace_back(residual[i]);
            }
            delete[] residual;  // clear meomery
            residual_by_pair[it->first] = result_by_cost_function;
        }
    }
};
/*
vertex_value | Dim:[D, Num]
 */
void GraphOptimizer::ResultOutput(Eigen::Ref<Eigen::MatrixXd> vertex_values, std::vector<int>& outside_ids)
{
    for(int i = 0; i < current_num_of_vertex_; i++) {
        for(int j = 0; j < dim_; j ++) {
            vertex_values(j, i) = vertex_values_[i * dim_ + j];
        }
    }

    outside_ids.clear();
    for(int i = 0; i < current_num_of_vertex_; i++){
        outside_ids.push_back(inside2outside_.at(i));
    }
};

void GraphOptimizer::ResultOutput(std::map<int, Eigen::MatrixXd>& output_values) {
    for(int i = 0; i < current_num_of_vertex_; i++) {
        int outside_id = inside2outside_.at(i);
        Eigen::MatrixXd output_matrix = Eigen::MatrixXd::Zero(4, 4);
        Quaternion2Matrix(&vertex_values_[i * dim_], output_matrix);
        output_values[outside_id] = output_matrix;
    }
};

// Tools for Quaternion
void GraphOptimizer::Quaternion2Matrix(const double* quaternion_pose, 
                                       Eigen::Ref<Eigen::MatrixXd> matrix) {
    double vectored_matrix[9];
    ceres::QuaternionToRotation(quaternion_pose, vectored_matrix); // row_major here
    
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            matrix(i, j) = vectored_matrix[i * 3 + j];
        }
    }

    // transition
    for(int i = 0; i < 3; i++) {
        matrix(i, 3) = quaternion_pose[4 + i];
    }

    // corner
    matrix(3, 3) = 1;
};

/*
quaternion_pose | Dim : 7
 */
void GraphOptimizer::Matrix2Quaternion(const Eigen::Ref<Eigen::MatrixXd>& matrix,
                                       double* quaternion_pose){
    double vectored_matrix[9];  // col major
    for(int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            vectored_matrix[j * 3 + i] = matrix(i, j);
        }
    }


    // rotation part
    ceres::RotationMatrixToQuaternion(vectored_matrix, quaternion_pose);

    // transition part
    for(int i = 0; i < 3; i++) {
        quaternion_pose[4 + i] = matrix(i, 3);
    }
};

void PoseDistance(const Eigen::Ref<Eigen::MatrixXd>& pose_1,
                  const Eigen::Ref<Eigen::MatrixXd>& pose_2,
                  const Eigen::Ref<Eigen::MatrixXd>& inner_transform,
                  std::vector<double>& distance_vector,
                  int symmetric_axis) {
    Eigen::MatrixXd relative_pose = pose_1.inverse() * pose_2;
    Eigen::MatrixXd transformed_rel_pose = 
        inner_transform.inverse() * relative_pose * inner_transform;
    // Transform into quaternion
    double rel_pose[7];
    GraphOptimizer::Matrix2Quaternion(relative_pose, rel_pose);
    if(symmetric_axis % 10 == 1) {
        // symmetric in z-axis
        distance_vector[0] = std::abs(rel_pose[1]) + std::abs(rel_pose[2]);
        distance_vector[1] = rel_pose[4];
        distance_vector[2] = rel_pose[5];
        distance_vector[3] = rel_pose[6];
    }
    else {
        if(rel_pose[0] > 0) {
            distance_vector[0] = std::abs(
                rel_pose[0] - 1.0
            );
        }
        else {
            distance_vector[0] = std::abs(
                rel_pose[0] + 1.0
            );
        }
        distance_vector[1] = rel_pose[4];
        distance_vector[2] = rel_pose[5];
        distance_vector[3] = rel_pose[6];
    }
};