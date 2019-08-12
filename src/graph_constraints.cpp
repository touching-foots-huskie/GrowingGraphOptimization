#include "graph_constraints.hpp"

ceres::CostFunction* DualPoseCost(const double* relative_pose, double weight){
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DualPoseCostFunctor,
        7, 7, 7>(new DualPoseCostFunctor(relative_pose, weight));
    return cost_function;
};

ceres::CostFunction* UniPoseCost(const double* object_pose_1,
                                 const double* relative_pose, double weight){

    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<UniPoseCostFunctor,
        7, 7>(new UniPoseCostFunctor(object_pose_1, relative_pose, weight));
    return cost_function;
};

// Geometry Relationship
ceres::CostFunction* DualPlane2PlaneCost(const double* normal_1, 
                                         const double* center_1,
                                         const double* normal_2,
                                         const double* center_2,
                                         double weight_1,
                                         double weight_2){

    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DualPlane2PlaneFunctor,
        2, 7, 7>(new DualPlane2PlaneFunctor(normal_1, center_1, normal_2, center_2, 
                                            weight_1, weight_2));
    
    return cost_function;
};

ceres::CostFunction* UniPlane2PlaneCost(const double* object_pose_1,
                                        const double* normal_1, 
                                        const double* center_1,
                                        const double* normal_2,
                                        const double* center_2,
                                        double weight_1, 
                                        double weight_2){

    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<UniPlane2PlaneFunctor,
        2, 7>(new UniPlane2PlaneFunctor(object_pose_1, 
                                        normal_1, center_1, normal_2, center_2, 
                                        weight_1, weight_2));
    
    return cost_function;
};

ceres::CostFunction* DualPlane2SurfCost(const double* normal_1, 
                                         const double* center_1,
                                         const double* normal_2,
                                         const double* center_2,
                                         double radius,
                                         double weight_1,
                                         double weight_2) {
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DualPlane2SurfFunctor,
        2, 7, 7>(new DualPlane2SurfFunctor(normal_1, center_1, normal_2, center_2, radius, 
                                           weight_1, weight_2));
    
    return cost_function;
};

ceres::CostFunction* UniPlane2SurfCost(const double* object_pose_1,
                                       const double* normal_1, 
                                       const double* center_1,
                                       const double* normal_2,
                                       const double* center_2,
                                       double radius,
                                       double weight_1,
                                       double weight_2){

    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<UniPlane2SurfFunctor,
        2, 7>(new UniPlane2SurfFunctor(object_pose_1, 
                                       normal_1, center_1, normal_2, center_2, radius,
                                       weight_1, weight_2));
    
    return cost_function;
};