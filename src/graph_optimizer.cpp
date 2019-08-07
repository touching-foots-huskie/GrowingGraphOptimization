#include "graph_optimizer.h"

// Set Num must be done before vertex adding
void GraphOptimizer::SetVertexNum(int num, int dim){
    assert(current_num_of_vertex_ == 0);
    dim_ = dim;
    total_num_of_vertex_ = num;
    vertex_values_.resize(num, dim);
};

// You'd better to set Vertex Num first
// Initial value Dim:[D, 1]
void GraphOptimizer::AddVertex(int outside_id, int local_param_type,
                               const Eigen::Ref<Eigen::MatrixXd>& initial_value){
    
    outside2inside_[outside_id] = current_num_of_vertex_;
    inside2outside_[current_num_of_vertex_] = outside_id;

    if(dim_ == 0){
        dim_ = initial_value.rows();
    }
    else{
        assert(initial_value.rows() == dim_);
    }
    
    if(current_num_of_vertex_ < current_num_of_vertex_){
        vertex_values_.block(0, current_num_of_vertex_, dim_, 1) = initial_value;
    }
    else {
        // expand total vertex value
        Eigen::MatrixXd pre_vertex_value = vertex_values_;
        vertex_values_.resize(dim_, 1 + vertex_values_.cols());
        vertex_values_ << pre_vertex_value, initial_value;
        total_num_of_vertex_ ++;
    }
    
    current_num_of_vertex_ ++;

    // add paramblock
    switch (local_param_type)
    {
    case 0:
        break;
    case 1:
        assert(dim_ == 6);
        // TODO: check if this pointer points to the vertex value
        problem_.AddParameterBlock(vertex_values_.block(0, current_num_of_vertex_, dim_, 1).data(), 
                                   dim_, new Sophus::test::LocalParameterizationSE3);
        break;
    default:
        std::cout << "No such local param" << std::endl;
        break;
    }
};

void GraphOptimizer::UpdateVertex(int outside_id, const Eigen::Ref<Eigen::MatrixXd>& update_value){
    int inside_id = outside2inside_[outside_id];
    vertex_values_.block(0, inside_id, dim_, 1) = update_value;
};

// Uni Edge only update data 1
void GraphOptimizer::AddUniEdge(int outside_id_1, int outside_id_2, 
                             ceres::CostFunction* cost_function_edge,
                             ceres::LossFunction* lost_function_edge){

    int inside_id_1 = outside2inside_[outside_id_1];
    int inside_id_2 = outside2inside_[outside_id_2];

    problem_.AddResidualBlock(cost_function_edge, lost_function_edge, 
                              vertex_values_.block(0, inside_id_1, dim_, 1).data());
};

// Dual Edge update both 1 & 2
void GraphOptimizer::AddDualEdge(int outside_id_1, int outside_id_2, 
                             ceres::CostFunction* cost_function_edge,
                             ceres::LossFunction* lost_function_edge){

    int inside_id_1 = outside2inside_[outside_id_1];
    int inside_id_2 = outside2inside_[outside_id_2];

    problem_.AddResidualBlock(cost_function_edge, lost_function_edge, 
                              vertex_values_.block(0, inside_id_1, dim_, 1).data(),
                              vertex_values_.block(0, inside_id_2, dim_, 1).data());
};

bool GraphOptimizer::Optimization(){
    ceres::Solve(options_, &problem_, &summary_);
    if((bool) optim_config_["report_output"]){
        std::cout << summary_.BriefReport() << "\n";
    }
    
    // TODO: decide if optimization is finished
    return true;
};

void GraphOptimizer::ResultOutput(Eigen::Ref<Eigen::MatrixXd> vertex_values, std::vector<int>& outside_ids)
{
    vertex_values = vertex_values_;
    outside_ids.clear();
    for(int i = 0; i < vertex_values_.cols(); i++){
        outside_ids.push_back(inside2outside_[i]);
    }
};
