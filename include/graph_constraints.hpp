#ifndef GRAPH_CONSTRAINTS_HPP
#define GRAPH_CONSTRAINTS_HPP

#include <ceres/ceres.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/rotation.h>
#include <Eigen/Eigen>

#define _USE_MATH_DEFINES
#include <math.h>

// Operation tools in Quternion
template<typename T> 
void QuatPoseCompose(const T* t1, const T* t12, T* t2) {
    // rotation
    ceres::QuaternionProduct(t1, t12, t2);
    // position
    ceres::QuaternionRotatePoint(t1, &t12[4], &t2[4]);
    for(int i = 0; i < 3; i++) {
        t2[i + 4] = t2[i + 4] + t1[i + 4];
    }
};

/*
get pose 2 in 1
 */
template<typename T> 
void QuatRelPose(const T* t1, const T* t2, T* t12) {
    T inverse_rotation[4];
    inverse_rotation[0] = t1[0];
    for(int i = 1; i < 4; i++) {
        inverse_rotation[i] = -t1[i];
    }
    ceres::QuaternionProduct(inverse_rotation, t2, t12);
    T transition2to1[3];
    for(int i = 0; i < 3; i++) {
        transition2to1[i] = t2[4 + i] - t1[4 + i];
    }
    ceres::QuaternionRotatePoint(inverse_rotation, transition2to1, &t12[4]);
};

/*
Relative Pose between 1 & 2, with 2 rotated for PI.
 */
template<typename T> 
void RelPoseParse(const T* object_pose_1, const T* object_pose_2,
                  const T* relative_pose_1, const T* relative_pose_2,
                  T* relative_pose_2in1) {

    // get global pose
    T global_pose_1[7];
    T global_pose_2[7];
    QuatPoseCompose(object_pose_1, relative_pose_1, global_pose_1);
    QuatPoseCompose(object_pose_2, relative_pose_2, global_pose_2);

    T r_global_pose_2[7];
    // Inverse quaternion : (Rotation z-axis for pi from x-axis)
    T rotate180[7] = {(T) 0.0, (T) 1.0, (T) 0.0, (T) 0.0, (T) 0.0, (T) 0.0, (T) 0.0};
    // rotate rotation2
    QuatPoseCompose(global_pose_2, rotate180, r_global_pose_2);
    // get relative rotation between the two transform
    QuatRelPose(global_pose_1, r_global_pose_2, relative_pose_2in1);
};

// Functors
struct DualPoseCostFunctor {
    /*
    relative_pose | Dim [7], (w, xr, yr, zr, xt, yt, zt)
    residual | Dim 4
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
        q1 * qr * q2-1 = I
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

        // rotation error
        T rotation_error = (estimated_rotation_2[0] * object_pose_2[0] + 
                            estimated_rotation_2[1] * object_pose_2[1] + 
                            estimated_rotation_2[2] * object_pose_2[2] + 
                            estimated_rotation_2[3] * object_pose_2[3]);
        // rotation error should be 1 or -1
        if(rotation_error > (T) 0.0) {
            residual[0] = casted_weight * (rotation_error - (T) 1.0);
        }
        else {
            residual[0] = casted_weight * (rotation_error + (T) 1.0);
        }

        T estimated_transition_2[3];
        ceres::QuaternionRotatePoint(object_pose_1, relative_transition, 
                                     estimated_transition_2);

        // transition residual
        for(int i = 0; i < 3; i++) {
            residual[1 + i] = (estimated_transition_2[i] - (object_pose_2[i + 4] - 
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

        // rotation error
        T rotation_error = (estimated_rotation_2[0] * object_pose_2[0] + 
                            estimated_rotation_2[1] * object_pose_2[1] + 
                            estimated_rotation_2[2] * object_pose_2[2] + 
                            estimated_rotation_2[3] * object_pose_2[3]);
                            
        // rotation error should be 1 or -1
        if(rotation_error > (T) 0.0) {
            residual[0] = casted_weight * (rotation_error - (T) 1.0);
        }
        else {
            residual[0] = casted_weight * (rotation_error + (T) 1.0);
        }

        T estimated_transition_2[3];
        ceres::QuaternionRotatePoint(object_pose_1, relative_transition, 
                                     estimated_transition_2);

        // transition residual
        for(int i = 0; i < 3; i++) {
            residual[1 + i] = (estimated_transition_2[i] - (object_pose_2[i + 4] - 
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

// Symmetric Pose Functor
struct DualSymmetricCostFunctor {
    /*
    relative_pose | Dim [7], (w, xr, yr, zr, xt, yt, zt)
    residual | Dim 4
    In this senario, poses are all symmetric with z-axis
     */
    
    DualSymmetricCostFunctor(const double* relative_pose, double weight)
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
        q1 * qr * q2-1 = I
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

        // rotation error on x_axis & y_axis
        T rotation_error_x = (- estimated_rotation_2[1] * object_pose_2[0] 
                              + estimated_rotation_2[0] * object_pose_2[1] 
                              - estimated_rotation_2[2] * object_pose_2[3] 
                              + estimated_rotation_2[3] * object_pose_2[2]);
        
        T rotation_error_y = (estimated_rotation_2[0] * object_pose_2[2] 
                              + estimated_rotation_2[1] * object_pose_2[3]
                              - estimated_rotation_2[2] * object_pose_2[0] 
                              - estimated_rotation_2[3] * object_pose_2[1]);

        residual[0] = casted_weight * rotation_error_x;
        residual[1] = casted_weight * rotation_error_y;

        T estimated_transition_2[3];
        ceres::QuaternionRotatePoint(object_pose_1, relative_transition, 
                                     estimated_transition_2);

        // transition residual
        for(int i = 0; i < 3; i++) {
            residual[2 + i] = (estimated_transition_2[i] - (object_pose_2[i + 4] - 
                object_pose_1[i + 4])) * casted_weight;
        }  
        return true;
    };

private:
    const double* relative_quaternion_;
    const double* relative_transition_;
    double weight_;
};

struct UniSymmetricCostFunctor {
    /*
    relative_pose | Dim [7], (w, xr, yr, zr, xt, yt, zt)
    residual | Dim 4
    In this senario, poses are all symmetric with z-axis
     */
    
    UniSymmetricCostFunctor(const double* object_pose_1, 
                            const double* relative_pose, double weight)
    {
        object_pose_1_ = object_pose_1;
        relative_quaternion_ = relative_pose;
        weight_ = weight;
        relative_transition_ = (const double *) &relative_pose[4];
    };

    template <typename T> 
    bool operator() (const T* object_pose_2, T* residual) const {
        /*
        In quaternion, we should have:
        q1 * qr = q2 : rotation residual
        q1 * qr * q2-1 = I
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

        // rotation error on x_axis & y_axis
        T rotation_error_x = (- estimated_rotation_2[1] * object_pose_2[0] 
                              + estimated_rotation_2[0] * object_pose_2[1] 
                              - estimated_rotation_2[2] * object_pose_2[3] 
                              + estimated_rotation_2[3] * object_pose_2[2]);
        
        T rotation_error_y = (estimated_rotation_2[0] * object_pose_2[2] 
                              + estimated_rotation_2[1] * object_pose_2[3]
                              - estimated_rotation_2[2] * object_pose_2[0] 
                              - estimated_rotation_2[3] * object_pose_2[1]);

        residual[0] = casted_weight * rotation_error_x;
        residual[1] = casted_weight * rotation_error_y;

        T estimated_transition_2[3];
        ceres::QuaternionRotatePoint(object_pose_1, relative_transition, 
                                     estimated_transition_2);

        // transition residual
        for(int i = 0; i < 3; i++) {
            residual[2 + i] = (estimated_transition_2[i] - (object_pose_2[i + 4] - 
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

// Geometry Functors Version1
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

// Geometry Functors Version 2
struct DualPlane2PlaneFunctorV2 {
    DualPlane2PlaneFunctorV2(const double* relative_pose_1, 
                             const double* relative_pose_2,
                             double weight_1,
                             double weight_2) : 
                        relative_pose_1_(relative_pose_1),
                        relative_pose_2_(relative_pose_2),
                        weight_r_(weight_1),
                        weight_t_(weight_2)
    {};

    template <typename T> 
    bool operator() (const T* object_pose_1, const T* object_pose_2, T* residual) const {
        // Standard transform
        // n1.dot(n2) = 0, (c1 - c2).dot(n1) = 0
        T casted_weight_r = (T) weight_r_;
        T casted_weight_t = (T) weight_t_;

        // cast pose 
        T relative_pose_1[7];
        T relative_pose_2[7];
        for(int i = 0; i < 7; i++) {
            relative_pose_1[i] = (T) relative_pose_1_[i];
            relative_pose_2[i] = (T) relative_pose_2_[i];
        }

        T relative_pose_2in1[7];
        // get relative rotation & position
        RelPoseParse(object_pose_1, object_pose_2, 
                     relative_pose_1, relative_pose_2,
                     relative_pose_2in1);

        // rotation error : rotation axis should be [0, 0, 1]
        residual[0] = casted_weight_r * relative_pose_2in1[1];
        residual[1] = casted_weight_r * relative_pose_2in1[2];
        // transition error
        residual[2] = casted_weight_t * relative_pose_2in1[6];
        
        return true;
    };

private:
    const double* relative_pose_1_;
    const double* relative_pose_2_;
    double  weight_r_;
    double  weight_t_;
};

// Geometry Functors Version 2
struct UniPlane2PlaneFunctorV2 {
    UniPlane2PlaneFunctorV2(const double* object_pose_1,
                            const double* relative_pose_1, 
                            const double* relative_pose_2,
                            double weight_1,
                            double weight_2) : 
                        object_pose_1_(object_pose_1),
                        relative_pose_1_(relative_pose_1),
                        relative_pose_2_(relative_pose_2),
                        weight_r_(weight_1),
                        weight_t_(weight_2)
    {};

    template <typename T> 
    bool operator() (const T* object_pose_2, T* residual) const {
        T object_pose_1[7];
        for(int i = 0; i < 7; i++) {
            object_pose_1[i] = (T) object_pose_1_[i];
        }

        // Standard transform
        // n1.dot(n2) = 0, (c1 - c2).dot(n1) = 0
        T casted_weight_r = (T) weight_r_;
        T casted_weight_t = (T) weight_t_;

        // cast pose 
        T relative_pose_1[7];
        T relative_pose_2[7];
        for(int i = 0; i < 7; i++) {
            relative_pose_1[i] = (T) relative_pose_1_[i];
            relative_pose_2[i] = (T) relative_pose_2_[i];
        }

        T relative_pose_2in1[7];
        // get relative rotation & position
        RelPoseParse(object_pose_1, object_pose_2, 
                     relative_pose_1, relative_pose_2,
                     relative_pose_2in1);

        // rotation error : rotation axis should be [0, 0, 1]
        residual[0] = casted_weight_r * relative_pose_2in1[1];
        residual[1] = casted_weight_r * relative_pose_2in1[2];
        // transition error
        residual[2] = casted_weight_t * relative_pose_2in1[6];
        
        return true;
    };

private:
    const double* object_pose_1_;
    const double* relative_pose_1_;
    const double* relative_pose_2_;
    double  weight_r_;
    double  weight_t_;
};

struct DualPlane2SurfFunctorV2 {
    DualPlane2SurfFunctorV2(const double* relative_pose_1, 
                            const double* relative_pose_2,
                            double distance,
                            double weight_1) : 
                        relative_pose_1_(relative_pose_1),
                        relative_pose_2_(relative_pose_2),
                        distance_(distance),
                        weight_t_(weight_1)
    {};

    template <typename T> 
    bool operator() (const T* object_pose_1, const T* object_pose_2, T* residual) const {
        // Standard transform
        // n1.dot(n2) = 0, (c1 - c2).dot(n1) = 0
        T casted_weight_t = (T) weight_t_;

        // cast pose 
        T relative_pose_1[7];
        T relative_pose_2[7];
        for(int i = 0; i < 7; i++) {
            relative_pose_1[i] = (T) relative_pose_1_[i];
            relative_pose_2[i] = (T) relative_pose_2_[i];
        }

        T relative_pose_2in1[7];
        // get relative rotation & position
        RelPoseParse(object_pose_1, object_pose_2, 
                     relative_pose_1, relative_pose_2,
                     relative_pose_2in1);

        // Two points should be at [0, 0, distance] : points1 & points2
        T boundary_points_1[3] = {(T) 0.0, (T) 0.0, (T) 1.0};
        T boundary_points_2[3] = {(T) 0.0, (T) 0.0, (T) -1.0};

        // rotate points
        T estimated_points_1[3];
        T estimated_points_2[3];
        ceres::QuaternionRotatePoint(relative_pose_2in1, boundary_points_1, estimated_points_1);
        ceres::QuaternionRotatePoint(relative_pose_2in1, boundary_points_2, estimated_points_2);

        // z axis
        T estimated_z = - relative_pose_2in1[6] + (T) distance_;
        residual[0] = casted_weight_t * (estimated_points_1[2] - estimated_z);
        residual[1] = casted_weight_t * (estimated_points_2[2] - estimated_z);
        
        return true;
    };

private:
    const double* relative_pose_1_;
    const double* relative_pose_2_;
    double distance_;
    double weight_t_;
};

struct UniPlane2SurfFunctorV2 {
    // Fix plane
    UniPlane2SurfFunctorV2(const double* object_pose_1,
                           const double* relative_pose_1, 
                           const double* relative_pose_2,
                           double distance,
                           double weight_1) : 
                        object_pose_1_(object_pose_1),
                        relative_pose_1_(relative_pose_1),
                        relative_pose_2_(relative_pose_2),
                        distance_(distance),
                        weight_t_(weight_1)
    {};

    template <typename T> 
    bool operator() (const T* object_pose_2, T* residual) const {
        T object_pose_1[7];
        for(int i = 0; i < 7; i++) {
            object_pose_1[i] = (T) object_pose_1_[i];
        }
        // Standard transform
        // n1.dot(n2) = 0, (c1 - c2).dot(n1) = 0
        T casted_weight_t = (T) weight_t_;

        // cast pose 
        T relative_pose_1[7];
        T relative_pose_2[7];
        for(int i = 0; i < 7; i++) {
            relative_pose_1[i] = (T) relative_pose_1_[i];
            relative_pose_2[i] = (T) relative_pose_2_[i];
        }

        T relative_pose_2in1[7];
        // get relative rotation & position
        RelPoseParse(object_pose_1, object_pose_2, 
                     relative_pose_1, relative_pose_2,
                     relative_pose_2in1);

        // Two points should be at [0, 0, distance] : points1 & points2
        T boundary_points_1[3] = {(T) 0.0, (T) 0.0, (T) 1.0};
        T boundary_points_2[3] = {(T) 0.0, (T) 0.0, (T) -1.0};

        // rotate points
        T estimated_points_1[3];
        T estimated_points_2[3];
        ceres::QuaternionRotatePoint(relative_pose_2in1, boundary_points_1, estimated_points_1);
        ceres::QuaternionRotatePoint(relative_pose_2in1, boundary_points_2, estimated_points_2);

        // z axis
        T estimated_z = - relative_pose_2in1[6] + (T) distance_;
        residual[0] = casted_weight_t * (estimated_points_1[2] - estimated_z);
        residual[1] = casted_weight_t * (estimated_points_2[2] - estimated_z);

        return true;
    };

private:
    const double* object_pose_1_;
    const double* relative_pose_1_;
    const double* relative_pose_2_;
    double distance_;
    double weight_t_;
};

struct UniSurf2PlaneFunctorV2 {
    // Fix surf
    UniSurf2PlaneFunctorV2(const double* object_pose_1,
                           const double* relative_pose_1, 
                           const double* relative_pose_2,
                           double distance,
                           double weight_1) : 
                        object_pose_2_(object_pose_1),
                        relative_pose_1_(relative_pose_2),
                        relative_pose_2_(relative_pose_1),
                        distance_(distance),
                        weight_t_(weight_1)
    {};

    template <typename T> 
    bool operator() (const T* object_pose_1, T* residual) const {
        T object_pose_2[7];
        for(int i = 0; i < 7; i++) {
            object_pose_2[i] = (T) object_pose_2_[i];
        }
        // Standard transform
        // n1.dot(n2) = 0, (c1 - c2).dot(n1) = 0
        T casted_weight_t = (T) weight_t_;

        // cast pose 
        T relative_pose_1[7];
        T relative_pose_2[7];
        for(int i = 0; i < 7; i++) {
            relative_pose_1[i] = (T) relative_pose_1_[i];
            relative_pose_2[i] = (T) relative_pose_2_[i];
        }

        T relative_pose_2in1[7];
        // get relative rotation & position
        RelPoseParse(object_pose_1, object_pose_2, 
                     relative_pose_1, relative_pose_2,
                     relative_pose_2in1);

        // Two points should be at [0, 0, distance] : points1 & points2
        T boundary_points_1[3] = {(T) 0.0, (T) 0.0, (T) 1.0};
        T boundary_points_2[3] = {(T) 0.0, (T) 0.0, (T) -1.0};

        // rotate points
        T estimated_points_1[3];
        T estimated_points_2[3];
        ceres::QuaternionRotatePoint(relative_pose_2in1, boundary_points_1, estimated_points_1);
        ceres::QuaternionRotatePoint(relative_pose_2in1, boundary_points_2, estimated_points_2);

        // z axis
        T estimated_z = - relative_pose_2in1[6] + (T) distance_;
        residual[0] = casted_weight_t * (estimated_points_1[2] - estimated_z);
        residual[1] = casted_weight_t * (estimated_points_2[2] - estimated_z);

        return true;
    };

private:
    const double* object_pose_2_;
    const double* relative_pose_1_;
    const double* relative_pose_2_;
    double distance_;
    double weight_t_;
};

// TODO: Functions
// Parameterization : Dim [7]
ceres::CostFunction* DualPoseCost(const double* relative_pose, double weight);
ceres::CostFunction* UniPoseCost(const double* object_pose_1,
                                 const double* relative_pose, double weight);
ceres::CostFunction* DualSymmetricCost(const double* relative_pose, double weight);
ceres::CostFunction* UniSymmetricCost(const double* object_pose_1,
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
                                        double weight_1,
                                        double weight_2);

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

// Geometry constraints Version2
ceres::CostFunction* DualPlane2PlaneCost(const double* relative_pose_1, 
                                         const double* relative_pose_2,
                                         double weight_1,
                                         double weight_2);

ceres::CostFunction* UniPlane2PlaneCost(const double* object_pose_1,
                                        const double* relative_pose_1, 
                                        const double* relative_pose_2,
                                        double weight_1,
                                        double weight_2);

// Plane2Surf
ceres::CostFunction* DualPlane2SurfCost(const double* relative_pose_1, 
                                        const double* relative_pose_2,
                                        double distance,
                                        double weight_1);

ceres::CostFunction* UniPlane2SurfCost(const double* object_pose_1,
                                       const double* relative_pose_1, 
                                       const double* relative_pose_2,
                                       double distance,
                                       double weight_1);


ceres::CostFunction* UniSurf2PlaneCost(const double* object_pose_1,
                                       const double* relative_pose_1, 
                                       const double* relative_pose_2,
                                       double distance,
                                       double weight_1);

// currently, we don't have this constraints               
ceres::CostFunction* UniSurf2PlaneCost();

// Surf2Surf
ceres::CostFunction* DualSurf2SurfCost();

// currently, we don't have this constraints 
ceres::CostFunction* UniSurf2SurfCost();

#endif
