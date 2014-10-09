#pragma once

class BALProblem {
public:
    ~BALProblem();

    int num_observations()       const;
    const double* observations() const;
    double* mutable_cameras();
    double* mutable_points();

    int num_cameras() const;
    int camera_index_for_observation(int i) const;

    double* mutable_camera_for_observation(int i);
    double* mutable_point_for_observation(int i);

    bool LoadFile(const char* filename);

    bool saveToPlyFile(const char* filename);

private:
    template<typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }

    template<typename T>
    void FprintfOrDie(FILE *fptr, const char *format, T value) {
        fprintf(fptr, format, value);
    }

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;

    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_;
};