#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/loss_function.h"
#include "include/saveMesh.hpp"
#include "include/ply.hpp"

const double WEIGHT_OF_Z_ERROR = 4;
const double WEIGHT_OF_TRIANGLE_GRADIENT_ERROR = 1;

struct PointZError {
    PointZError(double observed_z)
            : observed_z(observed_z) {
    }

    template<typename T>
    bool operator()(const T *const z, T *residuals) const {
        residuals[0] = z[0] - T(observed_z);
        residuals[0] *= T(WEIGHT_OF_Z_ERROR);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_z) {
        return (new ceres::AutoDiffCostFunction<PointZError, 1, 1>(new PointZError(observed_z)));
    }

    double observed_z;
};

struct FaceGradientError {
    FaceGradientError(double squareXY)
            : squareXY(squareXY) {
    }

    template<typename T>
    bool operator()(const T *const z1, const T *const z2, const T *const z3,
            T *residuals) const {
        const T *const z[] = {z1, z2, z3};
        for (int i = 0; i < 3; i++) {
            residuals[i] = z[i][0] - (z[(i + 1) % 3][0] + z[(i + 2) % 3][0]) / T(2);
            residuals[i] *= T(WEIGHT_OF_TRIANGLE_GRADIENT_ERROR);
        }
        return true;
    }

    static ceres::CostFunction *Create(double **points) {
        double dx13 = points[0][0] - points[2][0];
        double dx23 = points[1][0] - points[2][0];
        double dy13 = points[0][1] - points[2][1];
        double dy23 = points[1][1] - points[2][1];
        double squareXY = (dx13 * dy23 - dx23 * dy13) / 2;

        return (new ceres::AutoDiffCostFunction<FaceGradientError, 3, 1, 1, 1>(new FaceGradientError(squareXY)));
    }

    double squareXY;
};

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc != 3) {
        std::cerr << "usage: simple_bundle_adjuster <input_ply_file> <output_ply_file>\n";
        return 1;
    }

    PlyMesh mesh = loadPlyBinary(argv[1]);

    // Create residuals for each observation in the bundle adjustment problem.
    ceres::Problem problem;
    ceres::LossFunction*huberLoss = new ceres::HuberLoss(1.0);
    for (int i = 0; i < mesh.pointsCount; ++i) {
        ceres::CostFunction *cost_function = PointZError::Create(mesh.pointsxyz[i * 3 + 2]);
        problem.AddResidualBlock(cost_function,
                huberLoss,
                &mesh.pointsxyz[i * 3 + 2]);
    }
    for (int i = 0; i < mesh.facesCount; ++i) {
        double *points[3];
        for (int j = 0; j < 3; j++) {
            points[j] = &mesh.pointsxyz[3 * mesh.faces[i * 3 + j]];
        }
        ceres::CostFunction *cost_function = FaceGradientError::Create(points);
        problem.AddResidualBlock(cost_function,
                NULL,
                points[0] + 2,
                points[1] + 2,
                points[2] + 2);
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_progress_to_stdout = true;
    options.update_state_every_iteration = true;
    options.num_threads = 3;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    savePlyBinary(mesh, argv[2]);
    return 0;
}