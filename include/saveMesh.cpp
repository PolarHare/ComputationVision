#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "saveMesh.hpp"

// Read a Bundle Adjustment in the Large dataset.

BALProblem::~BALProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
}

int BALProblem::num_observations() const {
    return num_observations_;
}

const double *BALProblem::observations() const {
    return observations_;
}

double *BALProblem::mutable_cameras() {
    return parameters_;
}

double *BALProblem::mutable_points() {
    return parameters_ + 9 * num_cameras_;
}

int BALProblem::num_cameras() const {
    return num_cameras_;
}

int BALProblem::camera_index_for_observation(int i) const {
    return camera_index_[i];
}

double *BALProblem::mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * 9;
}

double *BALProblem::mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
}

bool BALProblem::LoadFile(const char *filename) {
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL) {
        return false;
    };

    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", camera_index_ + i);
        FscanfOrDie(fptr, "%d", point_index_ + i);
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
        }
    }

    for (int i = 0; i < num_parameters_; ++i) {
        FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    return true;
}

bool BALProblem::saveToPlyFile(const char *filename) {
    FILE *fptr = fopen(filename, "w");
    if (fptr == NULL) {
        return false;
    };

    FprintfOrDie(fptr, "%s\n", "ply");
    FprintfOrDie(fptr, "%s\n", "format ascii 1.0");
    FprintfOrDie(fptr, "element vertex %d\n", num_points_);
    FprintfOrDie(fptr, "%s\n", "property float32 x");
    FprintfOrDie(fptr, "%s\n", "property float32 y");
    FprintfOrDie(fptr, "%s\n", "property float32 z");
    FprintfOrDie(fptr, "%s\n", "end_header");

    int ind = 0;
    for (int i = 9 * num_cameras_; i < num_parameters_; ++i) {
        FprintfOrDie(fptr, "%f ", parameters_[i]);
        ind = (ind + 1) % 3;
        if (ind == 0) {
            FprintfOrDie(fptr, "%s\n", "");
        }
    }
    return true;
}