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
        delete vertex_values_;
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
        delete vertex_values_;
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
    int inside_id = outside2inside_[outside_id];
    for(int i = 0; i < dim_; i++) {
        vertex_values_[inside_id * dim_ + i] = update_value(i, 0);
    }

};

void GraphOptimizer::UpdateVertexQuat(int outside_id, const Eigen::Ref<Eigen::MatrixXd>& update_value){
    int inside_id = outside2inside_[outside_id];
    Matrix2Quaternion(update_value, &vertex_values_[inside_id * dim_]);
};

// Uni Edge only update data 1
void GraphOptimizer::AddUniEdge(int outside_id_1, int outside_id_2, 
                             ceres::CostFunction* cost_function_edge,
                             ceres::LossFunction* lost_function_edge){

    int inside_id_1 = outside2inside_[outside_id_1];

    problem_.AddResidualBlock(cost_function_edge, lost_function_edge, 
                              &vertex_values_[inside_id_1 * dim_]);
};

// Dual Edge update both 1 & 2
void GraphOptimizer::AddDualEdge(int outside_id_1, int outside_id_2, 
                             ceres::CostFunction* cost_function_edge,
                             ceres::LossFunction* lost_function_edge){

    int inside_id_1 = outside2inside_[outside_id_1];
    int inside_id_2 = outside2inside_[outside_id_2];

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
        outside_ids.push_back(inside2outside_[i]);
    }
};

void GraphOptimizer::ResultOutput(std::map<int, Eigen::MatrixXd>& output_values) {
    for(int i = 0; i < current_num_of_vertex_; i++) {
        int outside_id = inside2outside_[i];
        Eigen::MatrixXd output_matrix = Eigen::MatrixXd::Zero(4, 4);
        Quaternion2Matrix(&vertex_values_[i * dim_], output_matrix);
        output_values[outside_id] = output_matrix;
    }
};

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