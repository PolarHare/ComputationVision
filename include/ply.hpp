#pragma once

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

    ~PlyMesh() {
        delete[] pointsxyz;
        delete[] faces;
    }
};

PlyMesh loadPlyBinary(const char *filename);

PlyMesh savePlyBinary(PlyMesh const& mesh, const char *filename);