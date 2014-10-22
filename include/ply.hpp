#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

class PlyMesh {

public:
    int pointsCount = -1;
    int facesCount = -1;
    //array of cooridates: x0, y0, z0, x1, y1, z1, ...
    double *pointsxyz;
    //indexes of points: i00, i01, i02, i10, i11, i12, ...
    unsigned int *faces;

    PlyMesh(int pointsCount, int facesCount) : pointsCount(pointsCount), facesCount(facesCount) {
        pointsxyz = new double[pointsCount * 3];
        faces = new unsigned int[facesCount * 3];
    }

    PlyMesh(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud);

    PlyMesh(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, pcl::PolygonMesh const &triangles);

    PlyMesh(const PlyMesh &other) {
        pointsCount = other.pointsCount;
        facesCount = other.facesCount;
        pointsxyz = new double[pointsCount * 3];
        faces = new unsigned int[facesCount * 3];
        for (int i = 0; i < pointsCount * 3; ++i) {
            pointsxyz[i] = other.pointsxyz[i];
        }
        for (int i = 0; i < facesCount * 3; ++i) {
            faces[i] = other.faces[i];
        }
    }

    PlyMesh &operator=(const PlyMesh &other) {
        pointsCount = other.pointsCount;
        facesCount = other.facesCount;
        pointsxyz = new double[pointsCount * 3];
        faces = new unsigned int[facesCount * 3];
        for (int i = 0; i < pointsCount * 3; ++i) {
            pointsxyz[i] = other.pointsxyz[i];
        }
        for (int i = 0; i < facesCount * 3; ++i) {
            faces[i] = other.faces[i];
        }
        return *this;
    }

    ~PlyMesh() {
        delete[] pointsxyz;
        delete[] faces;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr toPclPointCloud() const;
};

PlyMesh loadPlyBinary(const char *filename);

void savePlyBinary(PlyMesh const &mesh, const char *filename);