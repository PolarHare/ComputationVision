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

using namespace cv;
using namespace std;

default_random_engine randomEngine(239);

void printPlot(char* name, vector<float> values, int width = 700, int height = 700) {
    int n = values.size();
    float minV = values[n - 1];
    float maxV = values[n - 1];
    for (float val : values) {
        minV = min(minV, val);
        maxV = max(maxV, val);
    }
    Scalar col(255, 100, 100);
    Mat plot = Mat::zeros(width, height, CV_32FC3);
    for (int i = 0; i < n; i++) {
        int fromX = i * width / n;
        int toX = (i + 1) * width / n;
        float normalized = values[i] - minV / maxV;
        rectangle(plot, Point2f(fromX, height), Point2f(toX, height * (1 - normalized)), col, FILLED);
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

Mat scaleToFit(Mat img, int targetHeight) {
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
    return pair<Size, Mat>(Size(maxX - minX, maxY - minY), Mat(3, 3, CV_32FC1, movement).clone());
}

enum RANSAC_TASK {
    HOMOGRAPHY_TASK
};

//X - point type
//M - model type
//createModelFoo - function, that creating model by given points
//rankingFoo - function of ranking correspondence of current point X to hypothetical model M
template<typename X, typename M>
pair<M, vector<bool>> ransacContrario(vector<X> points, int minPointsCount,
        function<M(vector<X>)> createModelFoo, function<float(M, X)> rankingFoo,
        RANSAC_TASK taskType, float wholeArea, int itersCount = 100) {

    assert (points.size() > minPointsCount);

    float maxInliers = 0;
    M bestModel;
    float bestThreshold;
    for (int iter = 0; iter < itersCount; iter++) {
        vector<X> curPoints;

        vector<int> indexes;
        for (int i = 0; i < points.size(); i++) {
            indexes.push_back(i);
        }
        shuffle(indexes.begin(), indexes.end(), randomEngine);
        for (int i = 0; i < minPointsCount; i++) {
            curPoints.push_back(points[indexes[i]]);
        }

        M model = createModelFoo(curPoints);
        vector<float> distances;
        for (int i = 0; i < points.size(); i++) {
            distances.push_back(rankingFoo(model, points[i]));
        }
        sort(distances.begin(), distances.end());

        int bestInliers;
        float bestThreshold;
        float minProbability = -1;
        cout << "Probs: ";//DEBUG
        vector<float> probs;
        for (int i = 4; i < distances.size(); i++) {
            float curThreshold = distances[i];
            int count = 0;
            for (float rank : distances) {
                if (rank <= curThreshold) {
                    count++;
                } else {
                    break;
                }
            }
            float probability;
            switch (taskType) {
                case (HOMOGRAPHY_TASK) : {
                    float r = curThreshold;
                    probability = (float) (count * (M_PI * r * r) / wholeArea);
                    break;
                }
                default : {
                    assert (false);
                }
            }
            cout << " " << count << "/" << probability;//DEBUG
            probs.push_back(probability);
            if (minProbability == -1 || probability < minProbability) {
                minProbability = probability;
                bestThreshold = curThreshold;
                bestInliers = count;
                cout << "!";//DEBUG
            }
        }
        printPlot("Probabilities", probs);//DEBUG
        waitKey();
        cout << endl;//DEBUG

        if (bestInliers > maxInliers) {
            maxInliers = bestInliers;
            bestModel = model;
        }
    }
    vector<bool> isInlier(points.size(), false);
    for (int i = 0; i < points.size(); i++) {
        X point = points[i];
        if (rankingFoo(bestModel, point) <= bestThreshold) {
            isInlier[i] = true;
        }
    }
    vector<X> inliers;
    for (int i = 0; i < points.size(); i++) {
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
    drawCirclesWithDist(keyPs[0], keyPs[1], goodMatches, isInlier, perspective12, move * H, move);

    warpPerspective(imgs[1], perspective21, move, perspective21.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    warpPerspective(imgs[0], perspective21, move * H, perspective21.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);
    drawCirclesWithDist(keyPs[0], keyPs[1], goodMatches, isInlier, perspective21, move * H, move, true);

    Mat withMatches;
    drawMatches(imgs[0], keyPs[0], imgs[1], keyPs[1], goodMatches, withMatches);

    imshow("With matches", scaleToFit(withMatches, 700));
    int SPACE_KEY = 1048608;
    while (true) {
        imshow("Perspective", scaleToFit(perspective12, 1000));
        if (waitKey() != SPACE_KEY) {
            break;
        }
        imshow("Perspective", scaleToFit(perspective21, 1000));
        if (waitKey() != SPACE_KEY) {
            break;
        }
    }
}