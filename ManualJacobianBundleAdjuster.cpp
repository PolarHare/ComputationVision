// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A minimal, self-contained bundle adjuster using Ceres, that reads
// files from University of Washington' Bundle Adjustment in the Large dataset:
// http://grail.cs.washington.edu/projects/bal
//
// This does not use the best configuration for solving; see the more involved
// bundle_adjuster.cc file for details.

#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "include/saveMesh.hpp"
#include "ceres/autodiff_cost_function.h"

using ceres::CostFunction;
using ceres::SizedCostFunction;
using ceres::Jet;

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError
        : public SizedCostFunction<2, 9, 3> {
public:
    SnavelyReprojectionError(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {
    }

    virtual bool Evaluate(double const *const *parameters,
            double *residuals,
            double **jacobians) const {
        double const *const camera = parameters[0];
        double const *const point = parameters[1];
        // camera[0,1,2] are the angle-axis rotation.
        double p[3];

        double data[9] = {0};
        ceres::MatrixAdapter<double, 3, 1> rotationMatrix(data);
        ceres::AngleAxisToRotationMatrix<double>(camera, rotationMatrix);

        double rotatedPoint[3] = {0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rotatedPoint[i] += data[3 * i + j] * point[j];
            }
            p[i] = rotatedPoint[i];
        }

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        double xp = -p[0] / p[2];
        double yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const double &l1 = camera[7];
        const double &l2 = camera[8];
        double r2 = xp * xp + yp * yp;
        double distortion = 1.0 + r2 * (l1 + l2 * r2);

        // Compute final projected point position.
        const double &focal = camera[6];
        double predicted_x = focal * distortion * xp;
        double predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

        if (jacobians != NULL
                && (jacobians[0] != NULL || jacobians[1] != NULL)) {
            Jet<double, 12> cameraJ[9];
            for (int i = 0; i < 9; i++) {
                cameraJ[i] = Jet<double, 12>(camera[i], i);
            }
            Jet<double, 12> pointJ[3];
            for (int i = 0; i < 3; i++) {
                pointJ[i] = Jet<double, 12>(point[i], 9 + i);
            }
            Jet<double, 12> pJ[3];
            ceres::AngleAxisRotatePoint(cameraJ, pointJ, pJ);

            pJ[0] += cameraJ[3];
            pJ[1] += cameraJ[4];
            pJ[2] += cameraJ[5];

            Jet<double, 12> xpJ = -pJ[0] / pJ[2];
            Jet<double, 12> ypJ = -pJ[1] / pJ[2];

            Jet<double, 12> l1J = cameraJ[7];
            Jet<double, 12> l2J = cameraJ[8];

            Jet<double, 12> r2J = xpJ * xpJ + ypJ * ypJ;
            Jet<double, 12> distortionJ = Jet<double, 12>(1.0) + r2J * (l1J + l2J * r2J);

            Jet<double, 12> focalJ = cameraJ[6];
            Jet<double, 12> predicted_xJ = focalJ * distortionJ * xpJ;
            Jet<double, 12> predicted_yJ = focalJ * distortionJ * ypJ;

            if (jacobians[0] != NULL) {
                for (int i = 0; i < 9; i++) {
                    jacobians[0][i] = predicted_xJ.v[i];
                    jacobians[0][9 + i] = predicted_yJ.v[i];
                }
            }

            if (jacobians[1] != NULL) {
                for (int i = 9; i < 12; i++) {
                    jacobians[1][i - 9] = predicted_xJ.v[i];
                    jacobians[1][3 + i - 9] = predicted_yJ.v[i];
                }
            }
        }

        return true;
    }

    double observed_x;
    double observed_y;
};

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc != 4) {
        std::cerr << "usage: simple_bundle_adjuster <bal_problem> <bal_problem_init_ply> <bal_problem_result_ply>\n";
        return 1;
    }

    BALProblem bal_problem;
    if (!bal_problem.LoadFile(argv[1])) {
        std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
        return 1;
    }

    bal_problem.saveToPlyFile(argv[2]);

    const double *observations = bal_problem.observations();

    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.

        ceres::CostFunction *cost_function = new SnavelyReprojectionError(observations[2 * i + 0], observations[2 * i + 1]);
        problem.AddResidualBlock(cost_function,
                NULL /* squared loss */,
                bal_problem.mutable_camera_for_observation(i),
                bal_problem.mutable_point_for_observation(i));
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    bal_problem.saveToPlyFile(argv[3]);
    return 0;
}