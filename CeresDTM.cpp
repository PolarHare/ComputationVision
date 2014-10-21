#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "include/ply.hpp"

const double WEIGHT_OF_Z_ERROR = 4;
const double WEIGHT_OF_TRIANGLE_GRADIENT_ERROR = 32;

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
    FaceGradientError(double dx12, double dx13, double dy12, double dy13)
            : dx12(dx12), dx13(dx13), dy12(dy12), dy13(dy13) {
    }

    template<typename T>
    bool operator()(const T *const z1, const T *const z2, const T *const z3, T *residuals) const {
        T dz12 = z2[0] - z1[0];
        T dz13 = z3[0] - z1[0];
        residuals[0] = dz12 * T(dy13) - dz13 * T(dy12);
        residuals[1] = dz13 * T(dx12) - dz12 * T(dx13);
        residuals[0] *= T(WEIGHT_OF_TRIANGLE_GRADIENT_ERROR);
        residuals[1] *= T(WEIGHT_OF_TRIANGLE_GRADIENT_ERROR);
        return true;
    }

    static ceres::CostFunction *Create(double **points) {
        double dx12 = points[1][0] - points[0][0];
        double dx13 = points[2][0] - points[0][0];
        double dy12 = points[1][1] - points[0][1];
        double dy13 = points[2][1] - points[0][1];
        return (new ceres::AutoDiffCostFunction<FaceGradientError, 2, 1, 1, 1>(new FaceGradientError(dx12, dx13, dy12, dy13)));
    }

    double dx12;
    double dx13;
    double dy12;
    double dy13;
};

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc != 3) {
        std::cerr << "usage: simple_bundle_adjuster <input_ply_file> <output_ply_file>\n";
        return 1;
    }

    PlyMesh mesh = loadPlyBinary(argv[1]);

    ceres::Problem problem;
    ceres::LossFunction *huberLoss = new ceres::HuberLoss(1.0);
    for (int i = 0; i < mesh.pointsCount; ++i) {
        ceres::CostFunction *cost_function = PointZError::Create(mesh.pointsxyz[i * 3 + 2]);
        problem.AddResidualBlock(cost_function,
                huberLoss,
                &mesh.pointsxyz[i * 3 + 2]);
    }
    ceres::LossFunction *facesLoss = new ceres::HuberLoss(1.0);
    for (int i = 0; i < mesh.facesCount; ++i) {
        double *points[3];
        for (int j = 0; j < 3; j++) {
            points[j] = &mesh.pointsxyz[3 * mesh.faces[i * 3 + j]];
        }
        ceres::CostFunction *cost_function = FaceGradientError::Create(points);
        problem.AddResidualBlock(cost_function,
                facesLoss,
                points[0] + 2,
                points[1] + 2,
                points[2] + 2);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.update_state_every_iteration = true;
    options.num_threads = 3;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    savePlyBinary(mesh, argv[2]);
    return 0;
}