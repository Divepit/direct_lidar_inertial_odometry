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
#include "nano_gicp/lsq_registration.h"

template class nano_gicp::LsqRegistration<PointType, PointType>;

namespace nano_gicp {

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::LsqRegistration() {
  this->reg_name_ = "LsqRegistration";
  max_iterations_ = 64;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;

  // Default backend for registration updates.
  // step_optimize() can dispatch to GN, but LM is the default and most robust path.
  lsq_optimizer_type_ = LSQ_OPTIMIZER_TYPE::LevenbergMarquardt;
  lm_debug_print_ = false;
  lm_max_iterations_ = 10;
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;

  final_hessian_.setIdentity();
  final_error_ = 0.;
}

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::~LsqRegistration() {}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setRotationEpsilon(double eps) {
  rotation_epsilon_ = eps;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setTransformationEpsilon(double eps) {
  transformation_epsilon_ = eps;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setMaximumIterations(int iter) {
  max_iterations_ = iter;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setInitialLambdaFactor(double init_lambda_factor) {
  lm_init_lambda_factor_ = init_lambda_factor;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setDebugPrint(bool lm_debug_print) {
  lm_debug_print_ = lm_debug_print;
}

template <typename PointTarget, typename PointSource>
const Eigen::Matrix<double, 6, 6>& LsqRegistration<PointTarget, PointSource>::getFinalHessian() const {
  return final_hessian_;
}

template <typename PointTarget, typename PointSource>
double LsqRegistration<PointTarget, PointSource>::getFinalError() const {
  return final_error_;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  // Sanitize the user-provided guess into a proper rigid transform.
  Eigen::Isometry3d x0 = Eigen::Isometry3d::Identity();
  x0.linear() =
    Eigen::Quaterniond(guess.template block<3, 3>(0, 0).template cast<double>())
      .normalized()
      .toRotationMatrix();
  x0.translation() = guess.template block<3, 1>(0, 3).template cast<double>();

  lm_lambda_ = -1.0;
  converged_ = false;

  if (lm_debug_print_) {
    std::cout << "********************************************" << std::endl;
    std::cout << "***************** optimize *****************" << std::endl;
    std::cout << "********************************************" << std::endl;
  }

  for (int i = 0; i < max_iterations_ && !converged_; i++) {
    nr_iterations_ = i;

    // One nonlinear least-squares step (LM or GN depending on configuration).
    Eigen::Isometry3d delta;
    if (!step_optimize(x0, delta)) {
      std::cerr << "optimization step failed to converge" << std::endl;
      break;
    }

    converged_ = is_converged(delta);
  }

  final_transformation_ = x0.cast<float>().matrix();
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::is_converged(const Eigen::Isometry3d& delta) const {
  const Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t = delta.translation();

  const Eigen::Matrix3d r_delta = (1.0 / rotation_epsilon_) * R.array().abs().matrix();
  const Eigen::Vector3d t_delta = (1.0 / transformation_epsilon_) * t.array().abs().matrix();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1.0;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::computeLinearizationAtGuess(
    const Matrix4& guess,
    Eigen::Matrix<double, 6, 6>& H,
    Eigen::Matrix<double, 6, 1>& b,
    double& error) {
  Eigen::Isometry3d x0 = Eigen::Isometry3d::Identity();
  x0.linear() =
      Eigen::Quaterniond(guess.template block<3, 3>(0, 0).template cast<double>())
          .normalized()
          .toRotationMatrix();
  x0.translation() = guess.template block<3, 1>(0, 3).template cast<double>();

  error = linearize(x0, &H, &b);
  H = 0.5 * (H + H.transpose());

  return std::isfinite(error) && H.allFinite() && b.allFinite();
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_optimize(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  // Central dispatch point for the nonlinear update method.
  switch (lsq_optimizer_type_) {
    case LSQ_OPTIMIZER_TYPE::LevenbergMarquardt:
      return step_lm(x0, delta);
    case LSQ_OPTIMIZER_TYPE::GaussNewton:
      return step_gn(x0, delta);
    default:
      return step_lm(x0, delta);
  }
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_gn(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  const double y0 = linearize(x0, &H, &b);

  // Keep the normal matrix numerically symmetric.
  H = 0.5 * (H + H.transpose());

  if (!std::isfinite(y0) || !H.allFinite() || !b.allFinite()) {
    delta.setIdentity();
    final_hessian_ = H;
    final_error_ = y0;
    return false;
  }

  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H);
  if (solver.info() != Eigen::Success) {
    delta.setIdentity();
    final_hessian_ = H;
    final_error_ = y0;
    return false;
  }

  const Eigen::Matrix<double, 6, 1> d = solver.solve(-b);
  if (solver.info() != Eigen::Success || !d.allFinite()) {
    delta.setIdentity();
    final_hessian_ = H;
    final_error_ = y0;
    return false;
  }

  delta.setIdentity();
  delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
  delta.translation() = d.tail<3>();

  const Eigen::Isometry3d x1 = delta * x0;
  const double y1 = compute_error(x1);
  if (!std::isfinite(y1)) {
    delta.setIdentity();
    final_hessian_ = H;
    final_error_ = y0;
    return false;
  }

  x0 = x1;
  final_hessian_ = H;
  final_error_ = y1;

  return true;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_lm(Eigen::Isometry3d& x0,
                                                        Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  const double y0 = linearize(x0, &H, &b);

  H = 0.5 * (H + H.transpose());

  if (!std::isfinite(y0) || !H.allFinite() || !b.allFinite()) {
    delta.setIdentity();
    final_hessian_ = H;
    final_error_ = y0;
    return false;
  }

  constexpr double kEps = 1e-12;
  Eigen::Matrix<double, 6, 1> s = H.diagonal().cwiseAbs().cwiseMax(kEps).cwiseSqrt();
  Eigen::Matrix<double, 6, 6> S = s.cwiseInverse().asDiagonal();

  Eigen::Matrix<double, 6, 6> Hs = S * H * S;
  Eigen::Matrix<double, 6, 1> bs = S * b;

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * Hs.diagonal().cwiseAbs().maxCoeff();
  }

  double nu = 2.0;
  constexpr double kLambdaMin = 1e-18;
  constexpr double kLambdaMax = 1e30;

  constexpr double kMaxRotStep   = 0.35;
  constexpr double kMaxTransStep = 0.30;

  const Eigen::Matrix<double, 6, 6> diagHs = Hs.diagonal().asDiagonal();

  for (int i = 0; i < lm_max_iterations_; ++i) {
    const Eigen::Matrix<double, 6, 6> A = Hs + lm_lambda_ * diagHs;

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& sing = svd.singularValues();
    Eigen::Matrix<double, 6, 6> S_inv = Eigen::Matrix<double, 6, 6>::Zero();
    const double rel = 1e-12 * std::max(1.0, sing(0));
    for (int k = 0; k < 6; ++k) {
      if (sing(k) > rel) {
        S_inv(k, k) = 1.0 / sing(k);
      }
    }
    const Eigen::Matrix<double, 6, 1> ds = svd.matrixV() * S_inv * svd.matrixU().transpose() * (-bs);

    if (!ds.allFinite()) {
      lm_lambda_ = std::min(kLambdaMax, std::max(2.0 * lm_lambda_, lm_lambda_ * nu));
      nu = std::min(2.0 * nu, 1e6);
      continue;
    }

    Eigen::Matrix<double, 6, 1> d = S * ds;

    const double rn = d.head<3>().norm();
    const double tn = d.tail<3>().norm();
    double fac = 1.0;
    if (rn > kMaxRotStep)   fac = std::min(fac, kMaxRotStep   / rn);
    if (tn > kMaxTransStep) fac = std::min(fac, kMaxTransStep / tn);
    if (fac < 1.0) d *= fac;

    delta.setIdentity();
    delta.linear()      = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();

    const Eigen::Isometry3d xi = delta * x0;
    const double yi = compute_error_frozen(xi);
    if (!std::isfinite(yi)) {
      lm_lambda_ = std::min(kLambdaMax, std::max(2.0 * lm_lambda_, lm_lambda_ * nu));
      nu = std::min(2.0 * nu, 1e6);
      continue;
    }

    const double Hd = (H * d).dot(d);
    const double pred = -d.dot(b) - 0.5 * Hd;

    double rho = -1.0;
    if (std::isfinite(pred) && pred > 0.0) {
      rho = (y0 - yi) / pred;
    }

    if (lm_debug_print_) {
      if (i == 0) {
        std::cout << boost::format(
          "--- LM optimization ---\n%5s %15s %15s %15s %15s %15s %5s\n")
          % "i" % "y0" % "yi" % "rho" % "lambda" % "|d|" % "dec";
      }
      const char dec = (rho > 0.0 && yi < y0) ? 'x' : ' ';
      std::cout << boost::format("%5d %15g %15g %15g %15g %15g %5c")
        % i % y0 % yi % rho % lm_lambda_ % d.norm() % dec << std::endl;
    }

    if (rho > 0.0 && yi < y0) {
      x0 = xi;
      lm_lambda_ = std::max(kLambdaMin,
                            lm_lambda_ * std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3)));
      nu = 2.0;
      final_hessian_ = H;
      final_error_   = yi;
      return true;
    }

    lm_lambda_ = std::min(kLambdaMax, lm_lambda_ * nu);
    nu = std::min(2.0 * nu, 1e6);

    if (!d.allFinite() || d.norm() < 1e-12) {
      delta.setIdentity();
      final_hessian_ = H;
      final_error_   = y0;
      return false;
    }
  }

  delta.setIdentity();
  final_hessian_ = H;
  final_error_   = y0;
  return false;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_lm_pcg(Eigen::Isometry3d& x0,
                                                        Eigen::Isometry3d& delta) {
  // Linearize at current x0
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  const double y0 = linearize(x0, &H, &b);

  // Symmetrize for numerical safety
  H = 0.5 * (H + H.transpose());

  if (!std::isfinite(y0) || !H.allFinite() || !b.allFinite()) {
    delta.setIdentity();
    final_hessian_ = H;
    final_error_ = y0;
    return false;
  }

  // Column scaling: S = diag(1 / sqrt(max(|Hii|, eps)))
  constexpr double kEps = 1e-12;
  const Eigen::Matrix<double, 6, 1> s =
    H.diagonal().cwiseAbs().cwiseMax(kEps).cwiseSqrt();
  const Eigen::Matrix<double, 6, 6> S = s.cwiseInverse().asDiagonal();

  // Solve in scaled coordinates
  const Eigen::Matrix<double, 6, 6> Hs = S * H * S;
  const Eigen::Matrix<double, 6, 1> bs = S * b;

  if (!Hs.allFinite() || !bs.allFinite()) {
    delta.setIdentity();
    final_hessian_ = H;
    final_error_ = y0;
    return false;
  }

  // Initialize LM lambda in scaled space
  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * Hs.diagonal().cwiseAbs().maxCoeff();
  }

  double nu = 2.0;
  constexpr double kLambdaMin = 1e-18;
  constexpr double kLambdaMax = 1e30;

  // Trust-region caps
  constexpr double kMaxRotStep   = 0.35;  // rad
  constexpr double kMaxTransStep = 0.30;  // m

  // PCG settings
  constexpr int    kPcgMaxIter = 12;
  constexpr double kPcgRelTol  = 1e-4;
  constexpr double kPcgAbsTol  = 1e-10;
  constexpr double kPrecReg    = 1e-12;

  const Eigen::Matrix<double, 6, 6> diagHs = Hs.diagonal().asDiagonal();
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

  // Hoist lambda-invariant blocks outside the inner LM loop.
  // diagHs is diagonal so its off-diagonal 3x3 blocks are zero:
  //   A.block<3,3>(0,3) = Hs.block<3,3>(0,3)  (lambda-invariant)
  //   A.block<3,3>(3,0) = Hs.block<3,3>(3,0)  (lambda-invariant)
  const Eigen::Matrix3d Hs_rr     = Hs.block<3,3>(0,0);
  const Eigen::Matrix3d Hs_tt     = Hs.block<3,3>(3,3);
  const Eigen::Matrix3d Hs_rt     = Hs.block<3,3>(0,3);
  const Eigen::Matrix3d Hs_tr     = Hs.block<3,3>(3,0);
  const Eigen::Matrix3d diagHs_rr = Hs.diagonal().head<3>().asDiagonal();
  const Eigen::Matrix3d diagHs_tt = Hs.diagonal().tail<3>().asDiagonal();

  auto invert_spd_3x3 = [&](const Eigen::Matrix3d& M) -> Eigen::Matrix3d {
    const Eigen::Matrix3d Msym = 0.5 * (M + M.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Msym);
    if (es.info() == Eigen::Success && es.eigenvectors().allFinite() && es.eigenvalues().allFinite()) {
      const Eigen::Vector3d evals = es.eigenvalues().cwiseMax(kPrecReg);
      return es.eigenvectors() * evals.cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
    }

    // Conservative fallback
    return (Msym + kPrecReg * I3).inverse();
  };

  for (int i = 0; i < lm_max_iterations_; ++i) {
    // Damped system in scaled coordinates:
    //   A ds = -bs
    // with A = Hs + lambda * diag(Hs)
    const Eigen::Matrix<double, 6, 6> A = Hs + lm_lambda_ * diagHs;
    if (!A.allFinite()) {
      lm_lambda_ = std::min(kLambdaMax, std::max(2.0 * lm_lambda_, lm_lambda_ * nu));
      nu = std::min(2.0 * nu, 1e6);
      continue;
    }

    // ------------------------------------------------------------------
    // Preconditioner from damped Schur-decoupled blocks
    // ------------------------------------------------------------------
    // Arr/Att vary with lambda; Art/Atr are lambda-invariant (hoisted).
    const Eigen::Matrix3d Arr = Hs_rr + lm_lambda_ * diagHs_rr;
    const Eigen::Matrix3d Att = Hs_tt + lm_lambda_ * diagHs_tt;

    const Eigen::Matrix3d Arr_reg = Arr + kPrecReg * I3;
    const Eigen::Matrix3d Att_reg = Att + kPrecReg * I3;

    const Eigen::Matrix3d Arr_reg_inv = invert_spd_3x3(Arr_reg);
    const Eigen::Matrix3d Att_reg_inv = invert_spd_3x3(Att_reg);

    Eigen::Matrix3d Prr = Arr - Hs_rt * Att_reg_inv * Hs_tr;
    Eigen::Matrix3d Ptt = Att - Hs_tr * Arr_reg_inv * Hs_rt;

    Prr = 0.5 * (Prr + Prr.transpose()) + kPrecReg * I3;
    Ptt = 0.5 * (Ptt + Ptt.transpose()) + kPrecReg * I3;

    const Eigen::Matrix3d Prr_inv = invert_spd_3x3(Prr);
    const Eigen::Matrix3d Ptt_inv = invert_spd_3x3(Ptt);

    auto apply_preconditioner =
      [&](const Eigen::Matrix<double, 6, 1>& r) -> Eigen::Matrix<double, 6, 1> {
        Eigen::Matrix<double, 6, 1> z;
        z.head<3>() = Prr_inv * r.head<3>();
        z.tail<3>() = Ptt_inv * r.tail<3>();
        return z;
      };

    // ------------------------------------------------------------------
    // PCG solve: A ds = -bs
    // ------------------------------------------------------------------
    const Eigen::Matrix<double, 6, 1> rhs = -bs;
    Eigen::Matrix<double, 6, 1> ds = Eigen::Matrix<double, 6, 1>::Zero();

    bool solved = false;
    const double rhs_norm = rhs.norm();
    const double tol = std::max(kPcgAbsTol, kPcgRelTol * rhs_norm);

    Eigen::Matrix<double, 6, 1> r = rhs;  // x=0 => r = rhs - A*x = rhs
    if (r.norm() <= tol) {
      solved = true;
    } else {
      Eigen::Matrix<double, 6, 1> z = apply_preconditioner(r);
      double rz_old = r.dot(z);

      if (std::isfinite(rz_old) && rz_old > 0.0 && z.allFinite()) {
        Eigen::Matrix<double, 6, 1> p = z;

        for (int k = 0; k < kPcgMaxIter; ++k) {
          const Eigen::Matrix<double, 6, 1> Ap = A * p;
          const double denom = p.dot(Ap);

          if (!std::isfinite(denom) || denom <= 0.0) {
            solved = false;
            break;
          }

          const double alpha = rz_old / denom;
          if (!std::isfinite(alpha)) {
            solved = false;
            break;
          }

          ds += alpha * p;
          r  -= alpha * Ap;

          if (!ds.allFinite() || !r.allFinite()) {
            solved = false;
            break;
          }

          if (r.norm() <= tol) {
            solved = true;
            break;
          }

          const Eigen::Matrix<double, 6, 1> z_new = apply_preconditioner(r);
          const double rz_new = r.dot(z_new);

          if (!z_new.allFinite() || !std::isfinite(rz_new) || rz_new <= 0.0) {
            solved = false;
            break;
          }

          const double beta = rz_new / rz_old;
          if (!std::isfinite(beta)) {
            solved = false;
            break;
          }

          p = z_new + beta * p;
          if (!p.allFinite()) {
            solved = false;
            break;
          }

          rz_old = rz_new;

          if (k == kPcgMaxIter - 1) {
            solved = (r.norm() <= tol);
          }
        }
      }
    }

    if (!solved || !ds.allFinite()) {
      // Increase lambda and retry
      lm_lambda_ = std::min(kLambdaMax, std::max(2.0 * lm_lambda_, lm_lambda_ * nu));
      nu = std::min(2.0 * nu, 1e6);
      continue;
    }

    // Undo scaling: d = S * ds
    Eigen::Matrix<double, 6, 1> d = S * ds;

    // Trust-region cap on original variables.
    // Keep ds consistent with d after capping.
    const double rn = d.head<3>().norm();
    const double tn = d.tail<3>().norm();
    double fac = 1.0;
    if (rn > kMaxRotStep)   fac = std::min(fac, kMaxRotStep   / rn);
    if (tn > kMaxTransStep) fac = std::min(fac, kMaxTransStep / tn);
    if (fac < 1.0) {
      d  *= fac;
      ds *= fac;
    }

    if (!d.allFinite() || !ds.allFinite()) {
      lm_lambda_ = std::min(kLambdaMax, std::max(2.0 * lm_lambda_, lm_lambda_ * nu));
      nu = std::min(2.0 * nu, 1e6);
      continue;
    }

    // Build delta in SE(3)
    delta.setIdentity();
    delta.linear()      = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();

    // Evaluate trial objective using frozen correspondences from linearize(x0).
    // Avoids a full k-NN pass for every rejected inner LM step.
    const Eigen::Isometry3d xi = delta * x0;
    const double yi = compute_error_frozen(xi);
    if (!std::isfinite(yi)) {
      lm_lambda_ = std::min(kLambdaMax, std::max(2.0 * lm_lambda_, lm_lambda_ * nu));
      nu = std::min(2.0 * nu, 1e6);
      continue;
    }

    // Predicted reduction from the SAME damped model actually solved:
    //   m(ds) = y0 + 2 bs^T ds + ds^T Hs ds + lambda ds^T diag(Hs) ds
    // so
    //   pred = y0 - m(ds)
    const double pred =
      -(2.0 * ds.dot(bs) + ds.dot(Hs * ds) + lm_lambda_ * ds.dot(diagHs * ds));

    double rho = -1.0;
    if (std::isfinite(pred) && pred > 0.0) {
      rho = (y0 - yi) / pred;
    }

    if (lm_debug_print_) {
      if (i == 0) {
        std::cout << boost::format(
          "--- LM optimization (PCG inner solve) ---\n%5s %15s %15s %15s %15s %15s %5s\n")
          % "i" % "y0" % "yi" % "rho" % "lambda" % "|d|" % "dec";
      }
      const char dec = (rho > 0.0 && yi < y0) ? 'x' : ' ';
      std::cout << boost::format("%5d %15g %15g %15g %15g %15g %5c")
        % i % y0 % yi % rho % lm_lambda_ % d.norm() % dec << std::endl;
    }

    if (rho > 0.0 && yi < y0) {
      // Accept
      x0 = xi;

      // Nielsen update
      lm_lambda_ = std::max(
        kLambdaMin,
        lm_lambda_ * std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3)));
      nu = 2.0;

      final_hessian_ = H;
      final_error_ = yi;
      return true;
    }

    // Reject: increase lambda and retry
    lm_lambda_ = std::min(kLambdaMax, lm_lambda_ * nu);
    nu = std::min(2.0 * nu, 1e6);

    if (d.norm() < 1e-12) {
      delta.setIdentity();
      final_hessian_ = H;
      final_error_ = y0;
      return false;
    }
  }

  delta.setIdentity();
  final_hessian_ = H;
  final_error_ = y0;
  return false;
}

}  // namespace nano_gicp
