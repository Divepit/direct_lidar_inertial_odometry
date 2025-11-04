/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

/***********************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2020, SMRT-AIST
 * All rights reserved.
 *************************************************************************/

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <memory>
#include <vector>

#include "nano_gicp/lsq_registration.h"

namespace nano_gicp {

// If not provided by the build, define a sensible default.
// (The .cc uses #ifndef too, so only one definition wins.)
#define NANO_GICP_PHTREE_MULT 1000

using CovarianceList =
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

template<typename PointSource, typename PointTarget>
class NanoGICP : public LsqRegistration<PointSource, PointTarget> {
public:
  using Scalar  = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource        = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr     = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr= typename PointCloudSource::ConstPtr;

  using PointCloudTarget        = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr     = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr= typename PointCloudTarget::ConstPtr;

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;

public:
  NanoGICP();
  ~NanoGICP() override;

  void setNumThreads(int n);
  void setCorrespondenceRandomness(int k);
  void setMaxCorrespondenceDistance(double corr);
  void setRegularizationMethod(RegularizationMethod method);

  void swapSourceAndTarget() override;
  void clearSource() override;
  void clearTarget() override;

  void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  void setSourceCovariances(const std::shared_ptr<const CovarianceList>& covs);
  void setInputTarget(const PointCloudTargetConstPtr& cloud) override;
  void setTargetCovariances(const std::shared_ptr<const CovarianceList>& covs);

  // Optional full (re)builders that also prep the PH-Tree
  void registerInputSource(const PointCloudSourceConstPtr& cloud);
  void registerInputTarget(const PointCloudTargetConstPtr& cloud);

  bool calculateSourceCovariances();
  bool calculateTargetCovariances();

  std::shared_ptr<const CovarianceList> getSourceCovariances() const { return source_covs_; }
  std::shared_ptr<const CovarianceList> getTargetCovariances() const { return target_covs_; }

  // -------- Delta-update API for target (submap) --------
  void   initEmptyTarget();
  int    appendTargetBlock(const PointCloudTarget& cloud_block,
                           const CovarianceList&   covs_block);
  void   eraseTargetRange(int base, int count);
  size_t targetSize() const;
  // ------------------------------------------------------

  void update_correspondences(const Eigen::Isometry3d& trans);

protected:
  void   computeTransformation(PointCloudSource& output, const Matrix4& guess) override;
  double linearize(const Eigen::Isometry3d& trans,
                   Eigen::Matrix<double, 6, 6>* H,
                   Eigen::Matrix<double, 6, 1>* b) override;
  double compute_error(const Eigen::Isometry3d& trans) override;

  template <typename PointT>
  bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
                             CovarianceList& covariances,
                             float& density);

public:
  std::shared_ptr<const CovarianceList> source_covs_;
  std::shared_ptr<const CovarianceList> target_covs_;

  float  source_density_ = 0.0f;
  float  target_density_ = 0.0f;
  int    num_correspondences = 0;

  // Helper: effective spatial resolution implied by ConverterMultiply.
  static constexpr int kPhTreeMultiplier = NANO_GICP_PHTREE_MULT;
  static constexpr double ph_tree_resolution_m() {
    return 1.0 / static_cast<double>(kPhTreeMultiplier);
  }

protected:
  int    num_threads_          = 1;
  int    k_correspondences_    = 10;
  double corr_dist_threshold_  = 0.1;
  RegularizationMethod regularization_method_ = RegularizationMethod::PLANE;

  CovarianceList        mahalanobis_;     // per-correspondence (RCR)^-1
  std::vector<int>      correspondences_; // size = input_->size()
  std::vector<float>    sq_distances_;    // squared Euclidean dists
};

}  // namespace nano_gicp
