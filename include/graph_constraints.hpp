#ifndef GRAPH_CONSTRAINTS_HPP
#define GRAPH_CONSTRAINTS_HPP

#include <ceres/ceres.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/rotation.h>
#include <Eigen/Eigen>

// Functors
struct DualPoseCostFunctor {
    /*
    relative_pose | Dim [7], (w, xr, yr, zr, xt, yt, zt)
     */
    
    DualPoseCostFunctor(const double* relative_pose, double weight)
    {
        weight_ = weight;
        relative_quaternion_ = relative_pose;
        relative_transition_ = (const double *) &relative_pose[4];

    };

    template <typename T> 
    bool operator() (const T* object_pose_1, const T* object_pose_2, T* residual) const {
        /*
        In quaternion, we should have:
        q1 * qr = q2 : rotation residual
        q1 * pr * q1-1  + p1 = p2
        q1 * pr * q1-1 = (p2 - p1) : transition residual
         */
        // cast to T
        T relative_quaternion[4];
        T relative_transition[3];
        T casted_weight = (T) weight_;

        for(int i = 0; i < 4; i++) {
            relative_quaternion[i] = (T) relative_quaternion_[i];
        }
        
        for(int i = 0; i < 3; i++) {
            relative_transition[i] = (T) relative_transition_[i];
        }

        T estimated_rotation_2[4];
        ceres::QuaternionProduct(object_pose_1, relative_quaternion, estimated_rotation_2);

        // rotation residual
        for(int i = 0; i < 4; i++) {
            residual[i] = (estimated_rotation_2[i] - object_pose_2[i]) * casted_weight;
        }

        T estimated_transition_2[3];
        ceres::QuaternionRotatePoint(object_pose_1, relative_transition, 
                                     estimated_transition_2);

        // transition residual
        for(int i = 0; i < 3; i++) {
            residual[4 + i] = (estimated_transition_2[i] - (object_pose_2[i + 4] - 
                object_pose_1[i + 4])) * casted_weight;
        }  
        return true;
    };

private:
    const double* relative_quaternion_;
    const double* relative_transition_;
    double weight_;
};

struct UniPoseCostFunctor {
    /*
    In UniPoseCost the 1st pose is fixed
    relative_pose | Dim [7], (w, xr, yr, zr, xt, yt, zt)
     */
    
    UniPoseCostFunctor(const double* object_pose_1, 
                       const double* relative_pose, double weight)
    {
        weight_ = weight;
        object_pose_1_ = object_pose_1;
        relative_quaternion_ = relative_pose;
        relative_transition_ = (const double *) &relative_pose[4];

    };

    template <typename T> 
    bool operator() (const T* object_pose_2, T* residual) const {
        /*
        In quaternion, we should have:
        q1 * qr = q2 : rotation residual
        q1 * pr * q1-1  + p1 = p2
        q1 * pr * q1-1 = (p2 - p1) : transition residual
         */
        // cast to T
        T object_pose_1[7];

        for(int i = 0; i < 7; i++) {
            object_pose_1[i] = (T) object_pose_1_[i];
        }

        T relative_quaternion[4];
        T relative_transition[3];
        T casted_weight = (T) weight_;

        for(int i = 0; i < 4; i++) {
            relative_quaternion[i] = (T) relative_quaternion_[i];
        }
        
        for(int i = 0; i < 3; i++) {
            relative_transition[i] = (T) relative_transition_[i];
        }

        T estimated_rotation_2[4];
        ceres::QuaternionProduct(object_pose_1, relative_quaternion, estimated_rotation_2);

        // rotation residual
        for(int i = 0; i < 4; i++) {
            residual[i] = (estimated_rotation_2[i] - object_pose_2[i]) * casted_weight;
        }

        T estimated_transition_2[3];
        ceres::QuaternionRotatePoint(object_pose_1, relative_transition, 
                                     estimated_transition_2);

        // transition residual
        for(int i = 0; i < 3; i++) {
            residual[4 + i] = (estimated_transition_2[i] - (object_pose_2[i + 4] - 
                object_pose_1[i + 4])) * casted_weight;
        }  
        return true;
    };

private:
    const double* object_pose_1_;
    const double* relative_quaternion_;
    const double* relative_transition_;
    double weight_;
};

struct DualPlane2PlaneFunctor {
    DualPlane2PlaneFunctor(const double* normal_1, 
                           const double* center_1,
                           const double* normal_2,
                           const double* center_2,
                           double weight_1,
                           double weight_2) : 
                        normal_1_(normal_1),
                        center_1_(center_1),
                        normal_2_(normal_2),
                        center_2_(center_2),
                        weight_r_(weight_1),
                        weight_t_(weight_2)
    {};

    template <typename T> 
    bool operator() (const T* object_pose_1, const T* object_pose_2, T* residual) const {
        // n1.dot(n2) = 0,, (c1 - c2).dot(n1) = 0
        // cast
        T casted_weight_r = (T) weight_r_;
        T casted_weight_t = (T) weight_t_;

        T casted_normal_1[3];
        T casted_normal_2[3];
        T casted_center_1[3];
        T casted_center_2[3];

        for(int i = 0; i < 3; i++) {
            casted_normal_1[i] = (T) normal_1_[i];
            casted_normal_2[i] = (T) normal_2_[i];
            casted_center_1[i] = (T) center_1_[i];
            casted_center_2[i] = (T) center_2_[i];
        }
        T estimated_normal_1[3];
        T estimated_normal_2[3];
        T estimated_center_1[3];
        T estimated_center_2[3];

        ceres::QuaternionRotatePoint(object_pose_1, casted_normal_1, estimated_normal_1);
        ceres::QuaternionRotatePoint(object_pose_2, casted_normal_2, estimated_normal_2);

        // direction error
        residual[0] = (T) 1.0 + ceres::DotProduct(estimated_normal_1, estimated_normal_2);
        residual[0] *= casted_weight_r;

        ceres::QuaternionRotatePoint(object_pose_1, casted_center_1, estimated_center_1);
        ceres::QuaternionRotatePoint(object_pose_2, casted_center_2, estimated_center_2);
        
        T diff_direction[3];
        for(int i = 0; i < 3; i++) {
            diff_direction[i] = (estimated_center_2[i] - estimated_center_1[i]) +
                (object_pose_2[i + 4] - object_pose_1[i + 4]);
        }

        residual[1] = ceres::DotProduct(estimated_normal_1, diff_direction);
        residual[1] *= casted_weight_t;
        return true;
    };

private:
    const double* normal_1_;
    const double* center_1_;
    const double* normal_2_;
    const double* center_2_;
    double  weight_r_;
    double  weight_t_;
};

struct UniPlane2PlaneFunctor {
    UniPlane2PlaneFunctor(const double* object_pose_1,
                          const double* normal_1, 
                          const double* center_1,
                          const double* normal_2,
                          const double* center_2,
                          double weight_1,
                          double weight_2) : object_pose_1_(object_pose_1),
                                        normal_1_(normal_1), center_1_(center_1),
                                        normal_2_(normal_2), center_2_(center_2), 
                                        weight_r_(weight_1), weight_t_(weight_2)
    {};

    template <typename T> 
    bool operator() (const T* object_pose_2, T* residual) const {   
        // n1.dot(n2) = 0,, (c1 - c2).dot(n1) = 0
        T object_pose_1[7];
        for(int i = 0; i < 7; i++) {
            object_pose_1[i] = (T) object_pose_1_[i];
        }
        // cast
        T casted_weight_r = (T) weight_r_;
        T casted_weight_t = (T) weight_t_;
        T casted_normal_1[3];
        T casted_normal_2[3];
        T casted_center_1[3];
        T casted_center_2[3];

        for(int i = 0; i < 3; i++) {
            casted_normal_1[i] = (T) normal_1_[i];
            casted_normal_2[i] = (T) normal_2_[i];
            casted_center_1[i] = (T) center_1_[i];
            casted_center_2[i] = (T) center_2_[i];
        }
        T estimated_normal_1[3];
        T estimated_normal_2[3];
        T estimated_center_1[3];
        T estimated_center_2[3];

        ceres::QuaternionRotatePoint(object_pose_1, casted_normal_1, estimated_normal_1);
        ceres::QuaternionRotatePoint(object_pose_2, casted_normal_2, estimated_normal_2);

        // direction error
        residual[0] = (T) 1.0 + ceres::DotProduct(estimated_normal_1, estimated_normal_2);
        residual[0] *= casted_weight_r;

        ceres::QuaternionRotatePoint(object_pose_1, casted_center_1, estimated_center_1);
        ceres::QuaternionRotatePoint(object_pose_2, casted_center_2, estimated_center_2);
        
        T diff_direction[3];
        for(int i = 0; i < 3; i++) {
            diff_direction[i] = (estimated_center_2[i] - estimated_center_1[i]) +
                (object_pose_2[i + 4] - object_pose_1[i + 4]);
        }

        // normalization
        residual[1] = ceres::DotProduct(estimated_normal_1, diff_direction);
        residual[1] *= casted_weight_t;
        return true;
    };

private:
    const double* object_pose_1_; 
    const double* normal_1_;
    const double* center_1_;
    const double* normal_2_;
    const double* center_2_;
    double weight_r_;
    double weight_t_;
};

struct DualPlane2SurfFunctor {
    DualPlane2SurfFunctor(const double* normal_1, 
                          const double* center_1,
                          const double* normal_2,
                          const double* center_2,
                          double radius,
                          double weight_1,
                          double weight_2):normal_1_(normal_1), center_1_(center_1),
                                        normal_2_(normal_2), center_2_(center_2), 
                                        radius_(radius), 
                                        weight_r_(weight_1), weight_t_(weight_2)
    {};

    template <typename T> 
    bool operator() (const T* object_pose_1, const T* object_pose_2, T* residual) const {
        // n1.dot(n2) = 0,, (c1 - c2).dot(n1) = 0
        // cast
        T casted_weight_r = (T) weight_r_;
        T casted_weight_t = (T) weight_t_;
        T casted_normal_1[3];
        T casted_normal_2[3];
        T casted_center_1[3];
        T casted_center_2[3];

        for(int i = 0; i < 3; i++) {
            casted_normal_1[i] = (T) normal_1_[i];
            casted_normal_2[i] = (T) normal_2_[i];
            casted_center_1[i] = (T) center_1_[i];
            casted_center_2[i] = (T) center_2_[i];
        }
        T estimated_normal_1[3];
        T estimated_normal_2[3];
        T estimated_center_1[3];
        T estimated_center_2[3];

        ceres::QuaternionRotatePoint(object_pose_1, casted_normal_1, estimated_normal_1);
        ceres::QuaternionRotatePoint(object_pose_2, casted_normal_2, estimated_normal_2);

        // direction error
        residual[0] = ceres::DotProduct(estimated_normal_1, estimated_normal_2);
        residual[0] *= casted_weight_r;

        ceres::QuaternionRotatePoint(object_pose_1, casted_center_1, estimated_center_1);
        ceres::QuaternionRotatePoint(object_pose_2, casted_center_2, estimated_center_2);
        
        T diff_direction[3];
        for(int i = 0; i < 3; i++) {
            diff_direction[i] = (estimated_center_2[i] - estimated_center_1[i]) +
                (object_pose_2[i + 4] - object_pose_1[i + 4]);
        }

        // In this case, estimated_distance should be postive
        T estimated_distance = ceres::DotProduct(estimated_normal_1, diff_direction);
        T theory_distance = (T) radius_;

        // distance error
        T distance_error = (theory_distance -  estimated_distance);
        residual[1] = distance_error * casted_weight_t ;
        return true;
    };

private:
    const double* normal_1_;
    const double* center_1_;
    const double* normal_2_;
    const double* center_2_;
    double radius_;
    double weight_r_;
    double weight_t_;
};

struct UniPlane2SurfFunctor {
    UniPlane2SurfFunctor(const double* object_pose_1,
                         const double* normal_1, 
                         const double* center_1,
                         const double* normal_2,
                         const double* center_2,
                         double radius,
                         double weight_1,
                         double weight_2) : object_pose_1_(object_pose_1),
                                        normal_1_(normal_1), center_1_(center_1),
                                        normal_2_(normal_2), center_2_(center_2), 
                                        radius_(radius), 
                                        weight_r_(weight_1), weight_t_(weight_2)
    {};

    template <typename T> 
    bool operator() (const T* object_pose_2, T* residual) const {
        // n1.dot(n2) = 0,, (c1 - c2).dot(n1) = 0   
        // cast
        T object_pose_1[7];

        for(int i = 0; i  < 7; i++) {
            object_pose_1[i] = (T) object_pose_1_[i];
        }

        T casted_weight_r = (T) weight_r_;
        T casted_weight_t = (T) weight_t_;
        T casted_normal_1[3];
        T casted_normal_2[3];
        T casted_center_1[3];
        T casted_center_2[3];

        for(int i = 0; i < 3; i++) {
            casted_normal_1[i] = (T) normal_1_[i];
            casted_normal_2[i] = (T) normal_2_[i];
            casted_center_1[i] = (T) center_1_[i];
            casted_center_2[i] = (T) center_2_[i];
        }
        T estimated_normal_1[3];
        T estimated_normal_2[3];
        T estimated_center_1[3];
        T estimated_center_2[3];

        ceres::QuaternionRotatePoint(object_pose_1, casted_normal_1, estimated_normal_1);
        ceres::QuaternionRotatePoint(object_pose_2, casted_normal_2, estimated_normal_2);

        // direction error
        residual[0] = ceres::DotProduct(estimated_normal_1, estimated_normal_2);
        residual[0] *= casted_weight_r;

        ceres::QuaternionRotatePoint(object_pose_1, casted_center_1, estimated_center_1);
        ceres::QuaternionRotatePoint(object_pose_2, casted_center_2, estimated_center_2);
        
        T diff_direction[3];
        for(int i = 0; i < 3; i++) {
            diff_direction[i] = (estimated_center_2[i] - estimated_center_1[i]) +
                (object_pose_2[i + 4] - object_pose_1[i + 4]);
        }

        T estimated_distance = ceres::DotProduct(estimated_normal_1, diff_direction);
        T theory_distance = (T) radius_;

        // distance error
        T distance_error = (theory_distance -  estimated_distance);
        residual[1] = distance_error * casted_weight_t ;
        return true;
    };

private:
    const double* object_pose_1_;
    const double* normal_1_;
    const double* center_1_;
    const double* normal_2_;
    const double* center_2_;
    double radius_;
    double weight_r_;
    double weight_t_;
};

// TODO: Functions
// Parameterization : Dim [7]
ceres::CostFunction* DualPoseCost(const double* relative_pose, double weight);
ceres::CostFunction* UniPoseCost(const double* object_pose_1,
                                 const double* relative_pose, double weight);

// Geometry constraints
// Plane2Plane
ceres::CostFunction* DualPlane2PlaneCost(const double* normal_1, 
                                         const double* center_1,
                                         const double* normal_2,
                                         const double* center_2,
                                         double weight_1,
                                         double weight_2);

ceres::CostFunction* UniPlane2PlaneCost(const double* object_pose_1,
                                        const double* normal_1, 
                                        const double* center_1,
                                        const double* normal_2,
                                        const double* center_2,
                                        double weight);

// Plane2Surf
ceres::CostFunction* DualPlane2SurfCost(const double* normal_1, 
                                        const double* center_1,
                                        const double* normal_2,
                                        const double* center_2,
                                        double radius,
                                        double weight_1,
                                        double weight_2);

ceres::CostFunction* UniPlane2SurfCost(const double* object_pose_1,
                                       const double* normal_1, 
                                       const double* center_1,
                                       const double* normal_2,
                                       const double* center_2,
                                       double radius,
                                       double weight_1,
                                       double weight_2);

// currently, we don't have this constraints               
ceres::CostFunction* UniSurf2PlaneCost();

// Surf2Surf
ceres::CostFunction* DualSurf2SurfCost();

// currently, we don't have this constraints 
ceres::CostFunction* UniSurf2SurfCost();

#endif