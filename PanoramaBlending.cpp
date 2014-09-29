#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc.hpp>
#include <assert.h>
#include <string>
#include <random>
#include <algorithm>
#include <math.h>
#include <dirent.h>

using namespace cv;
using namespace std;

default_random_engine randomEngine(239);

float logCnk10(int n, int k) {
    assert (n >= k && n >= 0 && k >= 0);
    float res = 0.0f;
    for (int i = 1; i <= k; i++) {
        res += log10(n - i + 1) - log(i);
    }
    return res;
}

void printPlot(const char *name, vector<float> values, int width = 700, int height = 700) {
    int offsetW = 2;
    int offsetH = 2;
    width -= 2 * offsetW;
    height -= 2 * offsetH;
    size_t n = values.size();
    float minV = values[n - 1];
    float maxV = values[n - 1];
    for (float val : values) {
        minV = min(minV, val);
        maxV = max(maxV, val);
    }
    Scalar col(255, 100, 100);
    Mat plot = Mat::zeros(width, height, CV_32FC3);
    for (int i = 0; i < n; i++) {
        size_t fromX = i * width / n;
        size_t toX = (i + 1) * width / n;
        float normalized = (values[i] - minV) / (maxV - minV);
        rectangle(plot, Point2f(offsetW + fromX, offsetH + height), Point2f(offsetW + toX, offsetH + height * (1 - normalized)), col, FILLED);
    }
    imshow(name, plot);
}

Point2f movePoint(Point2f p, Mat m) {
    float data[] = {p.x, p.y, 1};
    Mat pMat(3, 1, CV_32FC1, data);
    Mat p2Mat = m * pMat;
    p2Mat *= 1 / p2Mat.at<float>(2, 0);
    return Point2f(p2Mat.at<float>(0, 0), p2Mat.at<float>(1, 0));
}

void drawCirclesWithDist(vector<KeyPoint> p1, vector<KeyPoint> p2, vector<DMatch> matches, vector<bool> inlier, Mat &img, Mat HAndMove, Mat move, bool fromSecond = false) {
    Scalar cols[] = {Scalar(255, 100, 100), Scalar(100, 100, 255)};
    for (int i = 0; i < matches.size(); i++) {
        DMatch match = matches[i];
        Point2f from = p1[match.queryIdx].pt;
        Point2f to = p2[match.trainIdx].pt;
        if (!fromSecond) {
            circle(img, movePoint(from, HAndMove), 10, cols[inlier[i] ? 0 : 1], 3, LINE_8);
        } else {
            circle(img, movePoint(to, move), 10, cols[inlier[i] ? 0 : 1], 3, LINE_8);
        }
        line(img, movePoint(from, HAndMove), movePoint(to, move), cols[inlier[i] ? 0 : 1], 1, LINE_8);
    }
}

Mat scaleToFit(Mat img, int targetHeight = 700) {
    float scale = targetHeight * 1.0f / img.rows;
    Mat resized;
    resize(img, resized, Size((int) (img.cols * scale), (int) (img.rows * scale)));
    return resized;
}

pair<Size, Mat> getPerspectiveSizeAndMovement(Size img1, Size img2, Mat H) {
    float img1Corners[] = {
            0, (float) img1.width, 0, (float) img1.width,
            0, 0, (float) img1.height, (float) img1.height,
            1, 1, 1, 1
    };
    Mat corners(3, 4, CV_32FC1, img1Corners);
    Mat img1CorsRes = H * corners;
    for (int i = 0; i < img1CorsRes.cols; i++) {
        img1CorsRes.col(i) *= 1.0f / img1CorsRes.at<float>(2, i);
    }
    float minX = 0;
    float minY = 0;
    float maxX = img2.width;
    float maxY = img2.height;
    for (int i = 0; i < img1CorsRes.cols; i++) {
        minX = min(minX, img1CorsRes.at<float>(0, i));
        maxX = max(maxX, img1CorsRes.at<float>(0, i));
        minY = min(minY, img1CorsRes.at<float>(1, i));
        maxY = max(maxY, img1CorsRes.at<float>(1, i));
    }
    float vx = -minX;
    float vy = -minY;
    float movement[] = {
            1, 0, vx,
            0, 1, vy,
            0, 0, 1
    };
    return pair<Size, Mat>(Size((int) (maxX - minX), (int) (maxY - minY)), Mat(3, 3, CV_32FC1, movement).clone());
}

enum RANSAC_TASK {
    HOMOGRAPHY_TASK
};

template<typename X>
vector<X> takeRandomValues(vector<X> values, int n) {
    vector<X> result;
    vector<int> indexes;
    for (int i = 0; i < values.size(); i++) {
        indexes.push_back(i);
    }
    shuffle(indexes.begin(), indexes.end(), randomEngine);
    for (int i = 0; i < n; i++) {
        result.push_back(values[indexes[i]]);
    }
    return result;
}

//X - point type
//M - model type
//createModelFoo - function, that creating model by given points
//rankingFoo - function of ranking correspondence of current point X to hypothetical model M
template<typename X, typename M>
pair<M, vector<bool>> ransacContrario(vector<X> points, int minPointsCount,
        function<M(vector<X>)> createModelFoo, function<float(M, X)> rankingFoo,
        RANSAC_TASK taskType, float wholeArea, int itersCount = 100) {

    int n = points.size();
    assert (n > minPointsCount);

    M bestModel;
    float bestThreshold = -1;
    int bestInliers = -1;
    for (int iter = 0; iter < itersCount; iter++) {
        M model = createModelFoo(takeRandomValues(points, minPointsCount));

        vector<float> distances;
        for (int i = 0; i < n; i++) {
            distances.push_back(rankingFoo(model, points[i]));
        }
        sort(distances.begin(), distances.end());

        float minProbability = FLT_MAX;
        float bestModelThreshold = -1;
        int bestModelInliers = -1;
        int inliers = 0;
        for (int i = 0; i < distances.size(); i++) {
            float threshold = distances[i];
            inliers++;
            float probability;
            switch (taskType) {
                case (HOMOGRAPHY_TASK) : {
                    float r = threshold;
                    float pi = (float) M_PI;
                    float p = pi * r * r / wholeArea;
                    if (p <= 1.0f / 4.0f) {
                        int k = inliers;
                        probability = logCnk10(n, k) + log10(p) * k + log10(1 - p) * (n - k);
                    } else {
                        probability = FLT_MAX;
                    }
                    break;
                }
                default : {
                    assert (false);
                }
            }
            if (probability < minProbability) {
                minProbability = probability;
                bestModelThreshold = threshold;
                bestModelInliers = inliers;
            }
        }

        if (bestModelInliers > bestInliers || bestInliers == -1) {
            bestModel = model;
            bestThreshold = bestModelThreshold;
            bestInliers = bestModelInliers;
        }
    }

    vector<bool> isInlier(n, false);
    for (int i = 0; i < n; i++) {
        X point = points[i];
        if (rankingFoo(bestModel, point) <= bestThreshold) {
            isInlier[i] = true;
        }
    }
    vector<X> inliers;
    for (int i = 0; i < n; i++) {
        if (isInlier[i]) {
            inliers.push_back(points[i]);
        }
    }
    return pair<M, vector<bool>>(createModelFoo(inliers), isInlier);
}

Mat findMyHomography(vector<Point2f> srcPoints, vector<Point2f> dstPoints) {
    assert (srcPoints.size() == dstPoints.size());
    assert (srcPoints.size() >= 4);
    Mat src(srcPoints);
    Mat dst(dstPoints);
    int n = src.rows;
    Mat A(2 * n, 9, CV_32FC1);
    for (int i = 0; i < n; i++) {
        float xi1 = src.at<float>(0, 2 * i);
        float yi = src.at<float>(0, 2 * i + 1);
        float xi2 = dst.at<float>(0, 2 * i);
        float yi2 = dst.at<float>(0, 2 * i + 1);
        float data[] = {0, 0, 0, -xi1, -yi, -1, yi2 * xi1, yi2 * yi, yi2,
                xi1, yi, 1, 0, 0, 0, -xi2 * xi1, -xi2 * yi, -xi2};
        Mat Ai(2, 9, CV_32FC1, data);
        Ai.row(0).copyTo(A.row(2 * i + 0));
        Ai.row(1).copyTo(A.row(2 * i + 1));
    }
    SVD svd(A, 4);
    return svd.vt.row(svd.vt.rows - 1).reshape(1, 3);
}

Mat drawCircles(vector<KeyPoint> ps, Mat img, Scalar color = Scalar(200, 100, 100), int radius = 4) {
    Mat res;
    img.copyTo(res);
    for (KeyPoint p : ps) {
        circle(res, p.pt, radius, color, radius / 2, LINE_8);
    }
    return res;
}

bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

vector<Mat> loadAllImages(string folder = ".", string extension = ".jpg") {
    vector<Mat> imgFiles;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(folder.data())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            printf("File: %s", ent->d_name);
            string fileName = string(ent->d_name);
            if (hasEnding(fileName, extension)) {
                imgFiles.push_back(imread(folder + string("/") + fileName));
                printf(" is image");
            }
            printf("\n");
        }
        closedir(dir);
    } else {
        perror((string("While opening folder ") + folder).data());
    }
    return imgFiles;
}

pair<vector<vector<KeyPoint>>, vector<Mat>> computeDescriptors(vector<Mat> imgs) {
    ORB orb(1000);
    vector<vector<KeyPoint>> keyPss;
    for (int i = 0; i < imgs.size(); i++) {
        keyPss.push_back(vector<KeyPoint>());
        orb.detect(imgs[i], keyPss[i]);
        cout << "Image" << i << " key points count: " << keyPss[i].size() << endl;
    }

    vector<Mat> descrs;
    for (int i = 0; i < imgs.size(); i++) {
        descrs.push_back(Mat());
        orb.compute(imgs[i], keyPss[i], descrs[i]);
    }
    return pair<vector<vector<KeyPoint>>, vector<Mat>>(keyPss, descrs);
}

const int minFeaturePoints = 5;

vector<DMatch> findMatches(Mat descrs1, Mat descrs2, int i = -1, int j = -1) {
    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descrs1, descrs2, matches, 2);

    vector<DMatch> goodMatches;
    for (vector<DMatch> match12 : matches) {
        if (match12[0].distance < match12[1].distance * 0.7) {
            goodMatches.push_back(match12[0]);
        }
    }
    cout << "Good matches/all matches for i=" << i << " and j=" << j << ": " << goodMatches.size() << "/" << matches.size() << endl;
    return goodMatches;
}

int main(int argc, char **argv) {
    vector<Mat> imgs = loadAllImages();

    pair<vector<vector<KeyPoint>>, vector<Mat>> pointsAndDescrs = computeDescriptors(imgs);
    vector<vector<KeyPoint>> keyPss = pointsAndDescrs.first;
    vector<Mat> descrs = pointsAndDescrs.second;

    unsigned long n = imgs.size();
    for (int i = 0; i < n; i++) {
        Mat withCircles = drawCircles(keyPss[i], imgs[i], Scalar(200, 100, 100), 10);
        imshow(to_string(i) + string(" with circles"), scaleToFit(withCircles, 200));
    }

    vector<DMatch> matches[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (keyPss[i].size() != 0
                    && keyPss[j].size() != 0) {
                matches[i][j] = findMatches(descrs[i], descrs[j], i, j);
                matches[j][i] = matches[i][j];
            }
        }
    }

    waitKey();
}