#include "ply.hpp"
#include "exception"
#include <iostream>
#include <endian.h>
#include <pcl/PolygonMesh.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr PlyMesh::toPclPointCloud() const {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < pointsCount; ++i) {
        cloud->push_back(pcl::PointXYZ((float) pointsxyz[3 * i + 0],
                (float) pointsxyz[3 * i + 1],
                (float) pointsxyz[3 * i + 2]));
    }
    return cloud;
}

PlyMesh::PlyMesh(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud) {
    pointsCount = pointCloud->points.size();
    facesCount = 0;
    pointsxyz = new double[pointsCount * 3];
    faces = new unsigned int[facesCount * 3];
    for (int i = 0; i < pointCloud->points.size(); i++) {
        pointsxyz[3 * i + 0] = pointCloud->points[i].x;
        pointsxyz[3 * i + 1] = pointCloud->points[i].y;
        pointsxyz[3 * i + 2] = pointCloud->points[i].z;
    }
}

PlyMesh::PlyMesh(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, pcl::PolygonMesh const &triangles) {
    pointsCount = pointCloud->points.size();
    facesCount = triangles.polygons.size();
    pointsxyz = new double[pointsCount * 3];
    faces = new unsigned int[facesCount * 3];
    for (int i = 0; i < pointCloud->points.size(); i++) {
        pointsxyz[3 * i + 0] = pointCloud->points[i].x;
        pointsxyz[3 * i + 1] = pointCloud->points[i].y;
        pointsxyz[3 * i + 2] = pointCloud->points[i].z;
    }
    for (int i = 0; i < triangles.polygons.size(); i++) {
        assert (triangles.polygons[i].vertices.size() == 3);
        for (int j = 0; j < triangles.polygons[i].vertices.size(); j++) {
            faces[3 * i + j] = triangles.polygons[i].vertices[j];
        }
    }
}

PlyMesh loadPlyBinary(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        throw -1;
    };

    int vertices = -1;
    int faces = -1;
    fscanf(fp, "ply\n");
    fscanf(fp, "format binary_little_endian 1.0\n");
    fscanf(fp, "element vertex %d\n", &vertices);
    fscanf(fp, "property float x\n");
    fscanf(fp, "property float y\n");
    fscanf(fp, "property float z\n");
    fscanf(fp, "element face %d\n", &faces);
    fscanf(fp, "property list uchar int vertex_indices\n");
    fscanf(fp, "end_header\n");
    PlyMesh mesh(vertices, faces);
    for (int i = 0; i < vertices; i++) {
        float tmp[3];
        fread(tmp, sizeof(float), 3, fp);
        for (int j = 0; j < 3; j++) {
            mesh.pointsxyz[i * 3 + j] = tmp[j];
        }
    }
    for (int i = 0; i < faces; i++) {
        unsigned char n = -1;
        fread(&n, sizeof(unsigned char), 1, fp);
        fread(mesh.faces + i * 3, sizeof(unsigned int), 3, fp);
        for(int j=0;j<3;j++){
            assert (mesh.faces[i*3+j] < vertices);
        }
    }
    fclose(fp);
    return mesh;
}

void savePlyBinary(PlyMesh const &mesh, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        throw -1;
    };

    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", mesh.pointsCount);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "element face %d\n", mesh.facesCount);
    fprintf(fp, "property list uchar int vertex_indices\n");
    fprintf(fp, "end_header\n");
    for (int i = 0; i < mesh.pointsCount; i++) {
        float tmp[3];
        for (int j = 0; j < 3; j++) {
            tmp[j] = (float) mesh.pointsxyz[i * 3 + j];
        }
        fwrite(tmp, sizeof(float), 3, fp);
    }
    for (int i = 0; i < mesh.facesCount; i++) {
        unsigned char n = 3;
        fwrite(&n, sizeof(unsigned char), 1, fp);
        fwrite(mesh.faces + i * 3, sizeof(unsigned int), 3, fp);
    }
    fclose(fp);
}