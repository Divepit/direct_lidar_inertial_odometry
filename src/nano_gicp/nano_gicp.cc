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
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include "dlio/dlio.h"
#include "nano_gicp/nano_gicp.h"

template class nano_gicp::NanoGICP<PointType, PointType>;

namespace nano_gicp {

constexpr bool kUseRobustKernel = true;

// Robust scale in Mahalanobis-norm units.
// Since s = e' M e = ||r_whitened||^2,
// delta is roughly the inlier threshold on ||r_whitened||.
constexpr double kRobustDelta = 2.5;

inline double robust_weight_cauchy(const double s) {
  if (!kUseRobustKernel) {
    return 1.0;
  }

  const double d2 = kRobustDelta * kRobustDelta;
  return 1.0 / (1.0 + s / d2);
}

inline double robust_rho_cauchy(const double s) {
  if (!kUseRobustKernel) {
    return s;
  }

  const double d2 = kRobustDelta * kRobustDelta;
  return d2 * std::log1p(s / d2);
}


template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::NanoGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 20;
  reg_name_ = "NanoGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();

  regularization_method_ = RegularizationMethod::PLANE;
  correspondences_precomputed_ = false;
}

template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::~NanoGICP() {}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setNumThreads(int n) {
#ifdef _OPENMP
  if (n <= 0) {
    num_threads_ = omp_get_max_threads();
  } else {
    num_threads_ = n;
  }
#else
  num_threads_ = (n <= 0) ? 1 : n;
#endif
}

template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::computeInitialHessianAtGuess(
    const Matrix4& guess,
    Eigen::Matrix<double, 6, 6>& H,
    Eigen::Matrix<double, 6, 1>& b,
    double& error) {
  if (source_covs_ == nullptr || source_covs_->size() != input_->size()) {
    if (!calculateSourceCovariances()) {
      return false;
    }
  }

  if (target_covs_ == nullptr || target_covs_->size() != target_->size()) {
    if (!calculateTargetCovariances()) {
      return false;
    }
  }

  const bool ok = LsqRegistration<PointSource, PointTarget>::computeLinearizationAtGuess(
      guess, H, b, error);
  if (ok) {
    // correspondences_ and mahalanobis_ are now valid at `guess`.
    // Signal linearize() to reuse them on the first align() iteration.
    correspondences_precomputed_ = true;
  }
  return ok;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setMaxCorrespondenceDistance(double corr) {
  corr_dist_threshold_ = corr;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  source_kdtree_.swap(target_kdtree_);
  source_covs_.swap(target_covs_);

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::clearSource() {
  input_.reset();
  source_kdtree_.reset();
  source_covs_.reset();
  source_density_ = 0.0f;

  correspondences_.clear();
  sq_distances_.clear();
  mahalanobis_.clear();
  correspondences_precomputed_ = false;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  target_kdtree_.reset();
  target_covs_.reset();
  target_density_ = 0.0f;

  correspondences_.clear();
  sq_distances_.clear();
  mahalanobis_.clear();
  correspondences_precomputed_ = false;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::registerInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::registerInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);

  std::shared_ptr<nanoflann::KdTreeFLANN<PointSource>> source_kdtree = std::make_shared<nanoflann::KdTreeFLANN<PointSource>>();
  source_kdtree->setInputCloud(cloud);
  source_kdtree_ = source_kdtree;

  source_covs_.reset();
  correspondences_precomputed_ = false;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);

  std::shared_ptr<nanoflann::KdTreeFLANN<PointTarget>> target_kdtree = std::make_shared<nanoflann::KdTreeFLANN<PointTarget>>();
  target_kdtree->setInputCloud(cloud);
  target_kdtree_ = target_kdtree;

  target_covs_.reset();
  correspondences_precomputed_ = false;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setSourceCovariances(const std::shared_ptr<const CovarianceList>& covs) {
  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setTargetCovariances(const std::shared_ptr<const CovarianceList>& covs) {
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::calculateSourceCovariances() {
  std::shared_ptr<CovarianceList> source_covs = std::make_shared<CovarianceList>();
  std::shared_ptr<float> source_density = std::make_shared<float>();
  bool ret = calculate_covariances(input_, *source_kdtree_, *source_covs, *source_density);
  source_covs_ = source_covs;
  source_density_ = *source_density;
  return ret;
}

template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::calculateTargetCovariances() {
  std::shared_ptr<CovarianceList> target_covs = std::make_shared<CovarianceList>();
  std::shared_ptr<float> target_density = std::make_shared<float>();
  bool ret = calculate_covariances(target_, *target_kdtree_, *target_covs, *target_density);
  target_covs_ = target_covs;
  target_density_ = *target_density;
  return ret;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  // Registration requires per-point local covariance estimates for both clouds.
  // Rebuild lazily when input/target changed or cache is invalid.
  if (source_covs_ == nullptr || source_covs_->size() != input_->size()) {
    calculateSourceCovariances();
  }
  if (target_covs_ == nullptr || target_covs_->size() != target_->size()) {
    calculateTargetCovariances();
  }

  // Delegate optimization to the common least-squares backend (LM/GN dispatch).
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_ != nullptr && source_covs_->size() == input_->size());
  assert(target_covs_ != nullptr && target_covs_->size() == target_->size());

  if (!input_ || !target_ || !target_kdtree_) {
    correspondences_.clear();
    sq_distances_.clear();
    mahalanobis_.clear();
    num_correspondences = 0;
    return;
  }

  const float corr_sq_threshold =
    static_cast<float>(corr_dist_threshold_ * corr_dist_threshold_);

  Eigen::Isometry3f trans_f = trans.cast<float>();

  correspondences_.assign(input_->size(), -1);
  sq_distances_.assign(input_->size(), std::numeric_limits<float>::infinity());
  mahalanobis_.resize(input_->size());

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < static_cast<int>(input_->size()); i++) {
    mahalanobis_[i].setZero();

    PointTarget pt;
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();

    // 1-NN correspondence in the target map.
    const int num_found = target_kdtree_->nearestKSearch(pt, 1, k_indices, k_sq_dists);
    if (num_found < 1) {
      continue;
    }

    sq_distances_[i] = k_sq_dists[0];
    if (!(k_sq_dists[0] < corr_sq_threshold)) {
      continue;
    }

    const int target_index = k_indices[0];
    correspondences_[i] = target_index;

    const auto& cov_A = (*source_covs_)[i];
    const auto& cov_B = (*target_covs_)[target_index];

    // GICP covariance model:
    //   C = C_B + R * C_A * R^T
    // and Mahalanobis weight M = C^{-1}.
    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    Eigen::LDLT<Eigen::Matrix4d> ldlt(RCR);
    if (ldlt.info() != Eigen::Success) {
      correspondences_[i] = -1;
      sq_distances_[i] = std::numeric_limits<float>::infinity();
      continue;
    }

    const Eigen::Matrix4d M = ldlt.solve(Eigen::Matrix4d::Identity());
    if (ldlt.info() != Eigen::Success || !M.allFinite()) {
      correspondences_[i] = -1;
      sq_distances_[i] = std::numeric_limits<float>::infinity();
      continue;
    }

    mahalanobis_[i] = M;
    mahalanobis_[i](3, 3) = 0.0;
  }

  num_correspondences = std::count_if(
    correspondences_.begin(),
    correspondences_.end(),
    [](int c) { return c >= 0; });
}

template <typename PointSource, typename PointTarget>
double NanoGICP<PointSource, PointTarget>::linearize(
    const Eigen::Isometry3d& trans,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b) {
  if (correspondences_precomputed_) {
    // Correspondences were built by computeInitialHessianAtGuess() at the same
    // pose — skip the redundant k-NN pass and consume the flag.
    correspondences_precomputed_ = false;
  } else {
    update_correspondences(trans);
  }

  // Objective:
  //   sum_i rho( e_i^T M_i e_i )
  // where e_i = p_i^target - T * p_i^source and M_i is the GICP Mahalanobis weight.
  //
  // Robustification is implemented with IRLS:
  //   s_i = e_i^T M_i e_i
  //   w_i = rho'(s_i)
  //
  // Then the Gauss-Newton / LM normal-equation contribution is approximated as:
  //   H += w_i * J_i^T M_i J_i
  //   b += w_i * J_i^T M_i e_i
  //
  // This is the standard cheap robustification used in least-squares registration.
  double sum_errors = 0.0;

  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    const double s_raw = error.transpose() * mahalanobis_[i] * error;
    if (!std::isfinite(s_raw)) {
      continue;
    }

    const double s = std::max(0.0, s_raw);
    const double w = robust_weight_cauchy(s);

    sum_errors += robust_rho_cauchy(s);

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    // First-order Jacobian wrt se(3) perturbation [w, t].
    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;

    Hs[omp_get_thread_num()] += w * Hi;
    bs[omp_get_thread_num()] += w * bi;
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
double NanoGICP<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  // Recompute correspondences at the trial transform before evaluating the cost.
  // This makes LM/GN acceptance use the objective induced by the current trial pose,
  // instead of reusing stale correspondences from the previous linearization point.
  update_correspondences(trans);

  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    const double s_raw = error.transpose() * mahalanobis_[i] * error;
    if (!std::isfinite(s_raw)) {
      continue;
    }

    const double s = std::max(0.0, s_raw);
    sum_errors += robust_rho_cauchy(s);
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
double NanoGICP<PointSource, PointTarget>::compute_error_frozen(const Eigen::Isometry3d& trans) {
  // Evaluate the GICP cost at `trans` using correspondences_ and mahalanobis_
  // from the most recent linearize() call, without running k-NN search.
  // Used by step_lm_pcg() inner loop to cheaply evaluate rejected trial steps.
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < static_cast<int>(input_->size()); i++) {
    const int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    const double s_raw = error.transpose() * mahalanobis_[i] * error;
    if (!std::isfinite(s_raw)) {
      continue;
    }

    sum_errors += robust_rho_cauchy(std::max(0.0, s_raw));
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
bool NanoGICP<PointSource, PointTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  const nanoflann::KdTreeFLANN<PointT>& kdtree,
  CovarianceList& covariances,
  float& density) {

  covariances.clear();
  density = 0.0f;

  if (!cloud || cloud->empty() || k_correspondences_ <= 0) {
    return false;
  }

  covariances.resize(cloud->size());

  const int requested_k = std::min<int>(k_correspondences_, static_cast<int>(cloud->size()));
  if (requested_k <= 0) {
    return false;
  }

  float sum_k_sq_distances = 0.0f;
  int density_samples = 0;

  std::vector<int> k_indices(requested_k);
  std::vector<float> k_sq_distances(requested_k);

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) reduction(+:sum_k_sq_distances, density_samples) firstprivate(k_indices, k_sq_distances)
  for (int i = 0; i < static_cast<int>(cloud->size()); i++) {
    const int num_found = kdtree.nearestKSearch(cloud->at(i), requested_k, k_indices, k_sq_distances);

    covariances[i].setZero();

    // Need at least 3 neighbors for a meaningful local covariance estimate.
    // Fall back to an isotropic covariance instead of using uninitialized data.
    if (num_found < 3) {
      covariances[i].template block<3, 3>(0, 0).setIdentity();
      continue;
    }

    if (num_found > 1) {
      const int normalization = ((num_found - 1) * num_found) / 2;
      if (normalization > 0) {
        sum_k_sq_distances +=
          std::accumulate(k_sq_distances.begin() + 1, k_sq_distances.begin() + num_found, 0.0f)
          / static_cast<float>(normalization);
        density_samples += 1;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_found);
    for (int j = 0; j < num_found; j++) {
      neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / static_cast<double>(num_found);

    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      const double lambda = 1e-3;
      const Eigen::Matrix3d C =
        cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      const Eigen::Matrix3d C_inv = C.inverse();

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        cov.block<3, 3>(0, 0),
        Eigen::ComputeFullU | Eigen::ComputeFullV);

      Eigen::Vector3d values;
      switch (regularization_method_) {
        default:
          std::cerr << "here must not be reached" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
      }

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) =
        svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }

  density = (density_samples > 0)
              ? (sum_k_sq_distances / static_cast<float>(density_samples))
              : 0.0f;

  return true;
}

}  // namespace nano_gicp
