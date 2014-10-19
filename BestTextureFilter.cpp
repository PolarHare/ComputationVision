#include <stdio.h>
#include <iostream>
#include <memory>
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <assert.h>
#include <string>
#include <random>
#include <algorithm>
#include <math.h>

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

void drawPlot(const char *name, vector<float> values, int width = 700, int height = 700) {
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

void showImagesWithSwitching(const char *name, Mat images[], int count, int width = 700) {
    int i = 0;
    int key;
    const int ESC_KEY = 1048603;
    const int SPACE_KEY = 1048608;
    do {
        imshow(name, scaleToFit(images[i], width));
        key = waitKey();
        char keyChar = (char) key;
        if (key == SPACE_KEY) {
            i = (i + 1) % count;
        } else if (keyChar - '0' >= 1 && keyChar - '0' <= min(9, count)) {
            i = keyChar - '1';
        }
    } while (key != ESC_KEY);

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

Mat mergeByBestContrast(Mat img1, Mat img2, Mat move, Mat homography, Size perspectiveSize);

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

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: MyAContrarioRansac <folderName>"
                << endl << "Where folderName - folder with 1.jpg and 2.jpg images inside." << endl;
        return EXIT_FAILURE;
    }

    Mat imgs[2];
    for (int i = 0; i < 2; i++) {
        string file = argv[1] + to_string(i + 1) + string(".jpg");
        imgs[1 - i] = imread(file, IMREAD_COLOR);
        if (imgs[1 - i].empty()) {
            cout << "Cannot load image: " << file << endl;
        }
        imshow(string("Img") + to_string(1 - i), scaleToFit(imgs[1 - i], 500));
    }

    ORB orb(2000);
    vector<KeyPoint> keyPs[2];
    Mat imgsWithKeyPs[2];
    for (int i = 0; i < 2; i++) {
        orb.detect(imgs[i], keyPs[i]);
        imgsWithKeyPs[i] = drawCircles(keyPs[i], imgs[i], Scalar(200, 100, 100), 10);
        cout << "Image" << i << " key points count: " << keyPs[i].size() << endl;
        imshow(string("Img") + to_string(i), scaleToFit(imgsWithKeyPs[i], 500));
    }

    Mat descrs[2];
    for (int i = 0; i < 2; i++) {
        orb.compute(imgs[i], keyPs[i], descrs[i]);
    }

    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descrs[0], descrs[1], matches, 2);
    cout << "Matches count: " << matches.size() << endl;

    vector<DMatch> goodMatches;
    for (vector<DMatch> match12 : matches) {
        if (match12[0].distance < match12[1].distance * 0.7) {
            goodMatches.push_back(match12[0]);
        }
    }
    cout << "Good matches: " << goodMatches.size() << endl;


    pair<Mat, vector<bool>> homographyWithInliers = ransacContrario<DMatch, Mat>(goodMatches, 4,
            [&keyPs](vector<DMatch> matches) -> Mat {
                vector<Point2f> srcPoints;
                vector<Point2f> dstPoints;
                for (DMatch match : matches) {
                    srcPoints.push_back(keyPs[0][match.queryIdx].pt);
                    dstPoints.push_back(keyPs[1][match.trainIdx].pt);
                }
                return findMyHomography(srcPoints, dstPoints);
            },
            [&keyPs](Mat h, DMatch match) -> float {
                Point2f p1 = keyPs[0][match.queryIdx].pt;
                Point2f p2 = keyPs[1][match.trainIdx].pt;
                float p1Data[] = {p1.x, p1.y, 1};
                Mat m(3, 1, CV_32FC1, p1Data);
                Mat p12m = h * m;
                p12m *= 1 / p12m.at<float>(2, 0);
                Point2f p12 = Point2f(p12m.at<float>(0, 0), p12m.at<float>(1, 0));
                float dist = (float) norm(p2 - p12);
                return dist;
            },
            HOMOGRAPHY_TASK,
            imgs[1].rows * imgs[1].cols
    );

    Mat H = homographyWithInliers.first;
    vector<bool> isInlier = homographyWithInliers.second;

    int inliersCount = 0;
    for (int i = 0; i < isInlier.size(); i++) {
        if (isInlier[i]) {
            inliersCount++;
        }
    }
    cout << "Inliers count: " << inliersCount << endl;

    pair<Size, Mat> perspectiveSAndMove = getPerspectiveSizeAndMovement(Size(imgs[0].cols, imgs[0].rows), Size(imgs[1].cols, imgs[1].rows), H);
    cout << "Perspective size: " << perspectiveSAndMove.first << endl;

    Mat perspective12(perspectiveSAndMove.first, CV_32FC1);
    Mat perspective21(perspectiveSAndMove.first, CV_32FC1);
    Mat move = perspectiveSAndMove.second;

    warpPerspective(imgs[0], perspective12, move * H, perspective12.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    warpPerspective(imgs[1], perspective12, move, perspective12.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);
    Mat perspective12WithMatches = perspective12.clone();
    drawCirclesWithDist(keyPs[0], keyPs[1], goodMatches, isInlier, perspective12WithMatches, move * H, move);

    warpPerspective(imgs[1], perspective21, move, perspective21.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    warpPerspective(imgs[0], perspective21, move * H, perspective21.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);
    Mat perspective21WithMatches = perspective21.clone();
    drawCirclesWithDist(keyPs[0], keyPs[1], goodMatches, isInlier, perspective21WithMatches, move * H, move, true);

    Mat withMatches;
    drawMatches(imgs[0], keyPs[0], imgs[1], keyPs[1], goodMatches, withMatches);

    imshow("Perspective 12", scaleToFit(perspective12WithMatches));
    imshow("Perspective 21", scaleToFit(perspective21WithMatches));

    Mat perspectiveWithBestPixels = mergeByBestContrast(imgs[0], imgs[1], move, H, perspectiveSAndMove.first);
    Mat imgsToShow[] = {perspectiveWithBestPixels, perspective12, perspective21};
    showImagesWithSwitching("Best contrast choosen", imgsToShow, 3);
}

Mat mergeByBestContrast(Mat img1, Mat img2, Mat move, Mat homography, Size perspectiveSize) {
    Mat imgs[] = {img1, img2};
    Mat imgsPersp[2];
    Mat imgsPerspGrey[2];
    Mat imgsContrastOnPanorama[2];
    Mat imgsContrastMasks[2];
    for (int i = 0; i < 2; i++) {
        Mat imgGrey;
        cvtColor(imgs[i], imgGrey, COLOR_RGB2GRAY);

        Mat imgBlured;
        GaussianBlur(imgGrey, imgBlured, Size(31, 31), 0, 0);

        Mat imgContrast = imgGrey - imgBlured;
        multiply(imgContrast, imgContrast, imgContrast);

        imgsContrastOnPanorama[i] = Mat(perspectiveSize, CV_32FC1);
        if (i == 0) {
            warpPerspective(imgContrast, imgsContrastOnPanorama[0], move * homography, perspectiveSize, INTER_LINEAR, BORDER_CONSTANT, 0);
            warpPerspective(imgs[i], imgsPersp[i], move * homography, perspectiveSize, INTER_LINEAR, BORDER_CONSTANT, 0);
            warpPerspective(imgGrey, imgsPerspGrey[i], move * homography, perspectiveSize, INTER_LINEAR, BORDER_CONSTANT, 0);
        } else {
            warpPerspective(imgContrast, imgsContrastOnPanorama[1], move, perspectiveSize, INTER_LINEAR, BORDER_CONSTANT, 0);
            warpPerspective(imgs[i], imgsPersp[i], move, perspectiveSize, INTER_LINEAR, BORDER_CONSTANT, 0);
            warpPerspective(imgGrey, imgsPerspGrey[i], move, perspectiveSize, INTER_LINEAR, BORDER_CONSTANT, 0);
        }
        GaussianBlur(imgsContrastOnPanorama[i], imgsContrastOnPanorama[i], Size(63, 63), 0, 0);

        imgsContrastMasks[i] = Mat(imgsContrastOnPanorama[i].size(), imgsContrastOnPanorama[i].type());

    }
    for (int x = 0; x < imgsContrastOnPanorama[0].cols; x++) {
        for (int y = 0; y < imgsContrastOnPanorama[0].rows; y++) {
            if (imgsPerspGrey[1].at<uchar>(y, x) == 0) {
                imgsContrastMasks[0].at<char>(y, x) = 255;
            } else if (imgsPerspGrey[0].at<uchar>(y, x) == 0) {
                imgsContrastMasks[1].at<char>(y, x) = 255;
            } else if (imgsContrastOnPanorama[0].at<char>(y, x) > imgsContrastOnPanorama[1].at<char>(y, x)) {
                imgsContrastMasks[0].at<char>(y, x) = 255;
            } else {
                imgsContrastMasks[1].at<char>(y, x) = 255;
            }
        }
    }
//    showImagesWithSwitching("test", imgsContrastMasks, 2);
    Mat perspective(perspectiveSize, CV_32FC1);
    for (int i = 0; i < 2; i++) {
        Mat imgPersp(perspectiveSize, CV_32FC1);
        imgsPersp[i].copyTo(perspective, imgsContrastMasks[i]);
    }
    return perspective;
}