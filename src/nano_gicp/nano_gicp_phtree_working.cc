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
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
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

// ---- PH-Tree additions start ----
#include <phtree/phtree.h>
#include <limits>
#include <numeric>
#include <algorithm>

namespace ipht = ::improbable::phtree;

namespace {

// Squared-Euclidean metric (avoids sqrt in KNN)
template <int DIM>
struct DistSqEuclidean {
  using KEY = ipht::PhPointD<DIM>;
  inline double operator()(const KEY& a, const KEY& b) const noexcept {
    double s = 0.0;
    for (int d = 0; d < DIM; ++d) {
      const double v = a[d] - b[d];
      s += v * v;
    }
    return s;  // squared distance
  }
};

// PH-Tree v16 KNN filter must expose IsNodeValid/IsEntryValid
struct NotSelfFilter {
  int self = -1;

  template <typename KEY>
  inline bool IsNodeValid(const KEY&, int) const noexcept {
    // No pruning at node level; return true to keep search simple/fast.
    return true;
  }
  template <typename KEY, typename VALUE>
  inline bool IsEntryValid(const KEY&, const VALUE& v) const noexcept {
    return v != self;
  }
};

// File-local PH-Trees tied to last input clouds.
std::shared_ptr<ipht::PhTreeD<3, int>> g_source_tree;
std::shared_ptr<ipht::PhTreeD<3, int>> g_target_tree;
pcl::PointCloud<PointType>::ConstPtr   g_source_cloud;
pcl::PointCloud<PointType>::ConstPtr   g_target_cloud;

// Precomputed PH keys for each cloud (avoid re-creating keys per query)
static std::vector<ipht::PhPointD<3>> g_source_pts;
static std::vector<ipht::PhPointD<3>> g_target_pts;

template <typename PointT>
void PrecomputePhPoints(const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
                        std::vector<ipht::PhPointD<3>>& out) {
  if (!cloud) { out.clear(); return; }
  out.resize(cloud->size());
  for (size_t i = 0; i < cloud->size(); ++i) {
    const auto& p = cloud->at(i);
    out[i] = ipht::PhPointD<3>{double(p.x), double(p.y), double(p.z)};
  }
}

template <typename PointT>
std::shared_ptr<ipht::PhTreeD<3, int>>
BuildPhTree(const typename pcl::PointCloud<PointT>::ConstPtr& cloud) {
  using Tree = ipht::PhTreeD<3, int>;
  auto tree = std::make_shared<Tree>();
  if (!cloud) return tree;
  const int N = static_cast<int>(cloud->size());
  for (int i = 0; i < N; ++i) {
    const auto& p = cloud->at(i);
    if (!pcl::isFinite(p)) continue;
    ipht::PhPointD<3> key{double(p.x), double(p.y), double(p.z)};
    tree->try_emplace(key, i);
  }
  return tree;
}

} // anonymous namespace
// ---- PH-Tree additions end ----

template class nano_gicp::NanoGICP<PointType, PointType>;

namespace nano_gicp {

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
}

template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::~NanoGICP() {}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setNumThreads(int n) {
  num_threads_ = n;
#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
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

  // Swap PH-Tree state as well
  std::swap(g_source_tree, g_target_tree);
  std::swap(g_source_cloud, g_target_cloud);
  std::swap(g_source_pts,  g_target_pts);

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::clearSource() {
  input_.reset();
  source_covs_.reset();
  g_source_tree.reset();
  g_source_cloud.reset();
  g_source_pts.clear();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  target_covs_.reset();
  g_target_tree.reset();
  g_target_cloud.reset();
  g_target_pts.clear();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::registerInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) return;
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::registerInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) return;
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) return;

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);

  auto source_kdtree = std::make_shared<nanoflann::KdTreeFLANN<PointSource>>();
  source_kdtree->setInputCloud(cloud);
  source_kdtree_ = source_kdtree;

  // Build PH-Tree + precompute keys
  g_source_cloud = cloud;
  g_source_tree  = BuildPhTree<PointSource>(cloud);
  PrecomputePhPoints<PointSource>(cloud, g_source_pts);

  source_covs_.reset();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) return;

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);

  auto target_kdtree = std::make_shared<nanoflann::KdTreeFLANN<PointTarget>>();
  target_kdtree->setInputCloud(cloud);
  target_kdtree_ = target_kdtree;

  // Build PH-Tree + precompute keys
  g_target_cloud = cloud;
  g_target_tree  = BuildPhTree<PointTarget>(cloud);
  PrecomputePhPoints<PointTarget>(cloud, g_target_pts);

  target_covs_.reset();
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
  auto source_covs = std::make_shared<CovarianceList>();
  auto source_density = std::make_shared<float>();
  bool ret = calculate_covariances(input_, *source_kdtree_, *source_covs, *source_density);
  source_covs_ = source_covs;
  source_density_ = *source_density;
  return ret;
}

template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::calculateTargetCovariances() {
  auto target_covs = std::make_shared<CovarianceList>();
  auto target_density = std::make_shared<float>();
  bool ret = calculate_covariances(target_, *target_kdtree_, *target_covs, *target_density);
  target_covs_ = target_covs;
  target_density_ = *target_density;
  return ret;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (!source_covs_ || source_covs_->size() != input_->size()) {
    calculateSourceCovariances();
  }
  if (!target_covs_ || target_covs_->size() != target_->size()) {
    calculateTargetCovariances();
  }
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_ && source_covs_->size() == input_->size());
  assert(target_covs_ && target_covs_->size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();

  const int N = static_cast<int>(input_->size());
  const float corr_sq = static_cast<float>(corr_dist_threshold_ * corr_dist_threshold_);

  correspondences_.resize(N);
  sq_distances_.resize(N);
  mahalanobis_.resize(N);

  // Ensure target tree exists (outside OMP)
  if (!g_target_tree && g_target_cloud) {
    g_target_tree = BuildPhTree<PointTarget>(g_target_cloud);
  }
  auto* tree = g_target_tree.get();
  auto  tgt  = g_target_cloud;

#pragma omp parallel for num_threads(num_threads_) schedule(static)
  for (int i = 0; i < N; ++i) {
    // Transform query point
    PointTarget pt;
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();
    const ipht::PhPointD<3> q{double(pt.x), double(pt.y), double(pt.z)};

    int   nn_idx = -1;
    float nn_d2  = std::numeric_limits<float>::infinity();

    if (tree && tgt) {
      auto it = tree->begin_knn_query(1, q, DistSqEuclidean<3>{});
      if (it != tree->end()) {
        nn_idx = *it;
        nn_d2  = static_cast<float>(it.distance()); // already squared
      }
    }

    sq_distances_[i]    = nn_d2;
    correspondences_[i] = (nn_idx >= 0 && nn_d2 < corr_sq) ? nn_idx : -1;

    if (correspondences_[i] < 0) {
      continue;
    }

    const int target_index = correspondences_[i];
    const auto& cov_A = (*source_covs_)[i];
    const auto& cov_B = (*target_covs_)[target_index];

    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    mahalanobis_[i] = RCR.inverse();
    mahalanobis_[i](3, 3) = 0.0f;
  }

  // Keep original semantics
  num_correspondences = std::count_if(correspondences_.begin(), correspondences_.end(),
                                      [](int c){ return c > 0; });
}

template <typename PointSource, typename PointTarget>
double NanoGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  update_correspondences(trans);

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(static)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) continue;

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += error.transpose() * mahalanobis_[i] * error;

    if (!H || !b) continue;

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;

    Hs[omp_get_thread_num()] += Hi;
    bs[omp_get_thread_num()] += bi;
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
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(static)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) continue;

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += error.transpose() * mahalanobis_[i] * error;
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
bool NanoGICP<PointSource, PointTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  const nanoflann::KdTreeFLANN<PointT>& /*kdtree unused*/,
  CovarianceList& covariances,
  float& density) {

  covariances.resize(cloud->size());
  const int N = static_cast<int>(cloud->size());
  const int K = std::max(1, k_correspondences_);
  float sum_k_sq_distances = 0.0f;

  // Resolve PH-Tree and precomputed keys once
  std::shared_ptr<ipht::PhTreeD<3,int>> tree;
  const std::vector<ipht::PhPointD<3>>* pts = nullptr;

  if (cloud.get() == g_source_cloud.get()) {
    if (!g_source_tree) g_source_tree = BuildPhTree<PointT>(cloud);
    tree = g_source_tree; pts = &g_source_pts;
  } else if (cloud.get() == g_target_cloud.get()) {
    if (!g_target_tree) g_target_tree = BuildPhTree<PointT>(cloud);
    tree = g_target_tree; pts = &g_target_pts;
  } else {
    // Rare fallback: temporary tree + local precompute
    tree = BuildPhTree<PointT>(cloud);
    static thread_local std::vector<ipht::PhPointD<3>> local_pts;
    PrecomputePhPoints<PointT>(cloud, local_pts);
    pts = &local_pts;
  }

#pragma omp parallel num_threads(num_threads_)
  {
    // Per-thread buffers
    std::vector<int>   idx; idx.reserve(K);
    std::vector<float> d2;  d2.reserve(K);
    Eigen::Matrix<double,4,Eigen::Dynamic> neighbors(4, K);

    float thread_sum = 0.0f;

#pragma omp for schedule(static) nowait
    for (int i = 0; i < N; ++i) {
      const ipht::PhPointD<3>& q = (*pts)[i];

      idx.clear(); d2.clear();
      // self first
      idx.push_back(i);
      d2.push_back(0.f);

      // fetch K-1 neighbors excluding self (distance already squared)
      if (K > 1) {
        auto it = tree->begin_knn_query(K-1, q, DistSqEuclidean<3>{}, NotSelfFilter{i});
        for (; it != tree->end() && static_cast<int>(idx.size()) < K; ++it) {
          idx.push_back(*it);
          d2.push_back(static_cast<float>(it.distance()));
        }
      }
      // pad if needed
      while (static_cast<int>(idx.size()) < K) { idx.push_back(i); d2.push_back(0.f); }

      const int normalization = ((K - 1) * (2 + K)) / 2;
      if (normalization > 0) {
        thread_sum += std::accumulate(d2.begin()+1, d2.end(), 0.0f) /
                      static_cast<float>(normalization);
      }

      // Build neighbor matrix
      for (int j = 0; j < K; ++j) {
        neighbors.col(j) = cloud->at(idx[j]).getVector4fMap().template cast<double>();
      }
      neighbors.colwise() -= neighbors.rowwise().mean().eval();
      Eigen::Matrix4d cov = neighbors * neighbors.transpose() / static_cast<double>(K);

      // Regularization (unchanged)
      if (regularization_method_ == RegularizationMethod::NONE) {
        covariances[i] = cov;
      } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
        const double lambda = 1e-3;
        Eigen::Matrix3d C = cov.block<3, 3>(0, 0) + lambda * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d C_inv = C.inverse();
        covariances[i].setZero();
        covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
      } else {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d values;
        switch (regularization_method_) {
          default: std::abort();
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
    } // for i

#pragma omp atomic
    sum_k_sq_distances += thread_sum;
  } // omp parallel

  density = (N > 0) ? (sum_k_sq_distances / static_cast<float>(N)) : 0.0f;
  return true;
}

}  // namespace nano_gicp
