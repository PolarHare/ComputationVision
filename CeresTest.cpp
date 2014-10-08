#include "ceres/ceres.h"
#include <random>
#include <vector>
#include <functional>
#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include "glog/logging.h"

using namespace cv;
using std::vector;
using std::cout;
using std::endl;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::CauchyLoss;

Mat drawPoints(Mat img, vector<Point2f> points, Scalar color,
        float fromX = -10, float toX = 10, float fromY = -10, float toY = 10) {
    Mat res;
    img.copyTo(res);
    for (Point2f p : points) {
        float imX = ((p.x - fromX) * img.cols) / (toX - fromX);
        float imY = img.rows - (p.y - fromY) * img.rows / (toY - fromY);
        Point2f imPoint(imX, imY);
        circle(res, imPoint, 3, color, 1);
    }
    return res;
}

Mat drawFunction(Mat img, std::function<float(float)> foo, Scalar color = Scalar(255, 255, 255),
        float fromX = -10, float toX = 10, float fromY = -10, float toY = 10) {
    Mat res;
    img.copyTo(res);

    float prevX = fromX;
    float prevY = foo(prevX);
    float imPrevX = 0;
    float imPrevY = img.rows - (prevY - fromY) * img.rows / (toY - fromY);
    for (int c = 0; c < img.cols; ++c) {
        float x = fromX + ((toX - fromX) * c / img.cols);
        float y = foo(x);
        float imX = c;
        float imY = img.rows - (y - fromY) * img.rows / (toY - fromY);
        line(res, Point2f(imPrevX, imPrevY), Point2f(imX, imY), color, 1);
        imPrevX = imX;
        imPrevY = imY;
    }
    return res;
}

Mat drawFunction(std::function<float(float)> foo, Scalar color = Scalar(255, 255, 255), int width = 700, int height = 500,
        float fromX = -10, float toX = 10, float fromY = -10, float toY = 10) {
    Mat img(height, width, CV_8UC3);
    return drawFunction(img, foo, color, fromX, toX, fromY, toY);
}

std::default_random_engine randEngine(239);

vector<Point2f> generateRandomPoints(int n = 100, float fromX = -10, float toX = 10, float fromY = -10, float toY = 10, float deviation = 0.1) {
    vector<Point2f> res;
    for (int i = 0; i < n; i++) {
        std::uniform_real_distribution<float> disX(fromX, toX);
        std::uniform_real_distribution<float> disY(fromY, toY);
        res.push_back(Point2f(disX(randEngine), disY(randEngine)));
    }
    return res;
}

vector<Point2f> generateNoisedPoints(std::function<float(float)> foo, int n = 100, float fromX = -10, float toX = 10, float deviation = 0.1) {
    vector<Point2f> res;
    float stepX = (toX - fromX) / n;
    for (float x = fromX; x < toX; x += stepX) {
        float y = foo(x);
        std::normal_distribution<float> normalDistr(y, deviation);
        float nearY = normalDistr(randEngine);
        res.push_back(Point2f(x, nearY));
    }
    return res;
}

double mySin(double x, double a, double b, double c) {
    return a * sin(b * x + c);
}

struct SinResidual {
    SinResidual(double x, double y)
            : x_(x), y_(y) {
    }

    template<typename T>
    bool operator()(const T *const p, T *residual) const {
        residual[0] = T(y_) - p[0] * sin(p[1] * T(x_) + p[2]);
        return true;
    }

private:
    const double x_;
    const double y_;
};

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    const int pointsCount = 1000000;
    const int pointsOutliersCount = 10000;

    float a = 4.2;
    float b = 0.7;
    float c = 2;
    std::function<float(float)> foo = [&a, &b, &c](float x) -> float {
        return mySin(x, a, b, c);
    };

    vector<Point2f> points = generateNoisedPoints(foo, pointsCount);
    Mat imgFooWithPoints = drawFunction(foo);
    imgFooWithPoints = drawPoints(imgFooWithPoints, points, Scalar(100, 100, 255));
    imshow("Target function with points", imgFooWithPoints);

    double cur[] = {1, 1, 1};
    Problem problem;
    for (int i = 0; i < points.size(); ++i) {
        CostFunction *cost_function =
                new AutoDiffCostFunction<SinResidual, 1, 3>(new SinResidual(points[i].x, points[i].y));
        problem.AddResidualBlock(cost_function, NULL, cur);
    }
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    cout << "Result a=" << cur[0] << " b=" << cur[1] << " c=" << cur[2] << endl;

    std::function<float(float)> resultFoo = [&cur](float x) -> float {
        return mySin(x, cur[0], cur[1], cur[2]);
    };
    Mat imgWithResultFoo = drawFunction(imgFooWithPoints, resultFoo, Scalar(255, 100, 100));
    imshow("Result function", imgWithResultFoo);


    vector<Point2f> randomData = generateRandomPoints(pointsOutliersCount);
    Mat imgFooWithAllPoints = drawPoints(imgFooWithPoints, randomData, Scalar(255, 100, 255));
    imshow("Target function with all points", imgFooWithAllPoints);
    vector<Point2f> allPoints;
    for (int i = 0; i < points.size(); ++i) {
        allPoints.push_back(points[i]);
    }
    for (int j = 0; j < randomData.size(); ++j) {
        allPoints.push_back(randomData[j]);
    }


    double cur2[] = {1, 1, 1};

    Problem problem2;
    for (int i = 0; i < allPoints.size(); ++i) {
        CostFunction *cost_function =
                new AutoDiffCostFunction<SinResidual, 1, 3>(new SinResidual(allPoints[i].x, allPoints[i].y));
        problem2.AddResidualBlock(cost_function, new CauchyLoss(0.5), cur2);
    }
    Solver::Options options2;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary2;
    Solve(options2, &problem2, &summary2);
    cout << "Result a=" << cur2[0] << " b=" << cur2[1] << " c=" << cur2[2] << endl;
    cout << summary2.FullReport() << endl;

    std::function<float(float)> resultFoo2 = [&cur2](float x) -> float {
        return mySin(x, cur2[0], cur2[1], cur2[2]);
    };
    Mat imgWithResultFoo2 = drawFunction(imgFooWithAllPoints, resultFoo2, Scalar(255, 100, 100));
    imshow("Result function with all points", imgWithResultFoo2);

    waitKey();
}