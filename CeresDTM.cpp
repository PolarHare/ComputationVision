#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "include/ply.hpp"

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

const double WEIGHT_OF_Z_ERROR = 4;
const double WEIGHT_OF_TRIANGLE_GRADIENT_ERROR = 32;
const double WEIGHT_OF_EDGE_GRADIENT_ERROR = 8;

ceres::LossFunction *POINTS_Z_ERROR_LOSS = new ceres::HuberLoss(1.0);
ceres::LossFunction *FACES_GRADIENT_ERROR_LOSS = new ceres::HuberLoss(1.0);
ceres::LossFunction *EDGES_DZ_LOSS = new ceres::HuberLoss(1.0);

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

struct EdgeDzError {
    EdgeDzError(double dx, double dy)
            : dx(dx), dy(dy) {
    }

    template<typename T>
    bool operator()(const T *const z1, const T *const z2, T *residuals) const {
        T dz = z2[0] - z1[0];
        residuals[0] = dz;
        residuals[0] *= T(WEIGHT_OF_EDGE_GRADIENT_ERROR);
        return true;
    }

    static ceres::CostFunction *Create(double **points) {
        double dx = points[1][0] - points[0][0];
        double dy = points[1][1] - points[0][1];
        return (new ceres::AutoDiffCostFunction<EdgeDzError, 1, 1, 1>(new EdgeDzError(dx, dy)));
    }

    double dx;
    double dy;
};

PlyMesh removeOutliers(PlyMesh const &mesh) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = mesh.toPclPointCloud();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(8);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_filtered);

    return PlyMesh(cloud_filtered);
}

PlyMesh triangulate(PlyMesh const &mesh) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = mesh.toPclPointCloud();

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setKSearch(20);
    n.compute(*normals);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud(cloud_with_normals);

    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    gp3.setSearchRadius(50);

    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4);
    gp3.setMinimumAngle(M_PI / 18);
    gp3.setMaximumAngle(2 * M_PI / 3);
    gp3.setNormalConsistency(false);

    gp3.setInputCloud(cloud_with_normals);
    gp3.setSearchMethod(tree2);
    gp3.reconstruct(triangles);
    return PlyMesh(cloud, triangles);
}

void addFacesGradientErrorBlock(PlyMesh const &mesh, ceres::Problem &problem);

void addPointsZErrorBlock(PlyMesh const &mesh, ceres::Problem &problem);

void addEdgesDzErrorBlock(PlyMesh const &mesh, ceres::Problem &problem);

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc != 3) {
        std::cerr << "usage: simple_bundle_adjuster <input_ply_file> <output_ply_file>\n";
        return 1;
    }

    PlyMesh mesh = loadPlyBinary(argv[1]);
//    mesh = removeOutliers(mesh);
//    savePlyBinary(mesh, "/home/polarnick/tmp/tmpNoOutliers.ply");
//    mesh = triangulate(mesh);
//    savePlyBinary(mesh, "/home/polarnick/tmp/tmpNoOutliersTriangulated.ply");
//    loadPlyBinary("/home/polarnick/tmp/tmpNoOutliersTriangulated.ply");

    ceres::Problem problem;
    addPointsZErrorBlock(mesh, problem);
//    addFacesGradientErrorBlock(mesh, problem);
    addEdgesDzErrorBlock(mesh, problem);

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

void addPointsZErrorBlock(PlyMesh const &mesh, ceres::Problem &problem) {
    for (int i = 0; i < mesh.pointsCount; ++i) {
        ceres::CostFunction *cost_function = PointZError::Create(mesh.pointsxyz[i * 3 + 2]);
        problem.AddResidualBlock(cost_function,
                POINTS_Z_ERROR_LOSS,
                &mesh.pointsxyz[i * 3 + 2]);
    }
}

void addFacesGradientErrorBlock(PlyMesh const &mesh, ceres::Problem &problem) {
    for (int i = 0; i < mesh.facesCount; ++i) {
        double *points[3];
        for (int j = 0; j < 3; j++) {
            points[j] = &mesh.pointsxyz[3 * mesh.faces[i * 3 + j]];
        }
        ceres::CostFunction *cost_function = FaceGradientError::Create(points);
        problem.AddResidualBlock(cost_function,
                FACES_GRADIENT_ERROR_LOSS,
                points[0] + 2,
                points[1] + 2,
                points[2] + 2);
    }
}

void addEdgesDzErrorBlock(PlyMesh const &mesh, ceres::Problem &problem) {
    for (int i = 0; i < mesh.facesCount; ++i) {
        double *points[4];
        for (int j = 0; j < 3; j++) {
            points[j] = &mesh.pointsxyz[3 * mesh.faces[i * 3 + j]];
        }
        points[3] = points[0];

        for (int j = 0; j < 3; j++) {
            ceres::CostFunction *cost_function = EdgeDzError::Create(points + j);
            problem.AddResidualBlock(cost_function,
                    EDGES_DZ_LOSS,
                    points[j] + 2,
                    points[j + 1] + 2);
        }
    }
}