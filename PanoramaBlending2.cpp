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
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;
using namespace std;

default_random_engine randomEngine(239);
const float maxOutliers = 0.25f;

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

Mat keypointsVectorToMat(vector<KeyPoint> kpss) {
    Mat ps(3, (int) kpss.size(), CV_32FC1);
    for (int i = 0; i < kpss.size(); i++) {
        ps.at<float>(0, i) = kpss[i].pt.x;
        ps.at<float>(1, i) = kpss[i].pt.y;
        ps.at<float>(2, i) = kpss[i].pt.y;
    }
    return ps;
}

vector<Point2f> pointsToVector(Mat ps) {
    vector<Point2f> res;
    for (int i = 0; i < ps.cols; i++) {
        ps.col(i) /= ps.at<float>(2, i);
        res.push_back(Point2f(ps.at<float>(0, i), ps.at<float>(1, i)));
    }
    return res;
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

void drawPlot(const char *name, vector<float> values, int width = 700, int height = 700, bool minIsZero = false) {
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
    if (minIsZero) {
        minV = 0;
    }
    Scalar col(255, 100, 100);
    Mat plot = Mat::zeros(height, width, CV_32FC3);
    for (int i = 0; i < n; i++) {
        size_t fromX = i * width / n;
        size_t toX = (i + 1) * width / n;
        float normalized = (values[i] - minV) / (maxV - minV);
        rectangle(plot, Point2f(offsetW + fromX, offsetH + height), Point2f(offsetW + toX, offsetH + height * (1 - normalized)), col, FILLED);
    }
    imshow(name, plot);
}

bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

vector<Mat> loadAllImages(int IMREAD_flag = IMREAD_COLOR, string folder = ".", string extension = ".jpg") {
    vector<Mat> imgFiles;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(folder.data())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            printf("File: %s", ent->d_name);
            string fileName = string(ent->d_name);
            if (hasEnding(fileName, extension)) {
                printf(" is image %lu", imgFiles.size());
                imgFiles.push_back(imread(folder + string("/") + fileName, IMREAD_flag));
            }
            printf("\n");
        }
        closedir(dir);
    } else {
        perror((string("While opening folder ") + folder).data());
    }
    return imgFiles;
}

Mat drawCircles(vector<KeyPoint> ps, Mat img, Scalar color = Scalar(200, 100, 100), int radius = 4, int tickness = 1) {
    Mat res;
    img.copyTo(res);
    for (KeyPoint p : ps) {
        circle(res, p.pt, radius, color, tickness, LINE_8);
    }
    return res;
}

Mat drawCircles(vector<Point2f> ps, Mat img, Scalar color = Scalar(200, 100, 100), int radius = 4, int tickness = 1) {
    Mat res;
    img.copyTo(res);
    for (Point2f p : ps) {
        circle(res, p, radius, color, tickness, LINE_8);
    }
    return res;
}

Mat scaleToFit(Mat img, int targetHeight = 700) {
    float scale = targetHeight * 1.0f / img.rows;
    Mat resized;
    resize(img, resized, Size((int) (img.cols * scale), (int) (img.rows * scale)));
    return resized;
}

pair<vector<vector<KeyPoint>>, vector<Mat>> computeDescriptors(vector<Mat> imgs, int points = 2000, bool silent = true) {
//    FAST
    xfeatures2d::SURF detector(3000);
//    ORB detector(points);
    vector<vector<KeyPoint>> keyPss;
    vector<Mat> descrs;

    for (int i = 0; i < imgs.size(); i++) {
        keyPss.push_back(vector<KeyPoint>());

        Mat grayImg;
        cvtColor(imgs[i], grayImg, COLOR_BGR2GRAY);
        imshow(string("gray") + to_string(i), scaleToFit(grayImg, 500));

        Mat img;
        createCLAHE(4, Size(imgs[i].cols / 100, imgs[i].rows / 100))->apply(grayImg, img);
//        equalizeHist(grayImg, grayImg);
        imshow(string("grayH") + to_string(i), scaleToFit(img, 500));
//        waitKey(10);
//
//        detector.detect(grayImg, keyPss[i]);
        detector.detect(img, keyPss[i]);
        if (!silent) {
            cout << "Image " << i << " key points count: " << keyPss[i].size() << endl;
        }

        descrs.push_back(Mat());
//        detector.compute(grayImg, keyPss[i], descrs[i]);
        detector.compute(imgs[i], keyPss[i], descrs[i]);
    }
    return pair<vector<vector<KeyPoint>>, vector<Mat>>(keyPss, descrs);
}

vector<DMatch> findMatches(Mat descrs1, Mat descrs2, float threshold = 0.75, int traceI = -1, int traceJ = -1) {
//    BFMatcher matcher(NORM_HAMMING);
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descrs1, descrs2, matches, 2);
    vector<DMatch> goodMatches;
    for (vector<DMatch> match12 : matches) {
        if (match12[0].distance < match12[1].distance * threshold) {
            goodMatches.push_back(match12[0]);
        }
    }
//    if (traceI != -1 && traceJ != -1) {
    cout << "Images " << traceI << "-" << traceJ << " have good matches/all matches" << ": " << goodMatches.size() << "/" << matches.size() << endl;
//    }
    return goodMatches;
}

struct ImgNode {
    int id;
    Mat componentImage;
    Mat toComponentTransformation;
    Mat img;
    vector<int> to = vector<int>();
    vector<Mat> revHomo = vector<Mat>();
    vector<Mat> transformations = vector<Mat>();

    ImgNode() {
    }

    ImgNode(int id, Mat const &toComponentTransformation, Mat const &img)
            : id(id),
              toComponentTransformation(toComponentTransformation),
              img(img) {
    }
};

void drawComponents(ImgNode nodes[], int n);

float logCnk10(int n, int k) {
    assert (n >= k && n >= 0 && k >= 0);
    float res = 0.0f;
    for (int i = 1; i <= k; i++) {
        res += log10(n - i + 1) - log(i);
    }
    return res;
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
    int outliersCount = count(isInlier.begin(), isInlier.end(), false);
    if (outliersCount > maxOutliers * points.size()) {
        return pair<M, vector<bool>>(Mat(), isInlier);
    } else {
        return pair<M, vector<bool>>(createModelFoo(inliers), isInlier);
    }
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

pair<Mat, vector<bool>> findHomographyACRansac(vector<DMatch> goodMatches, vector<KeyPoint> ps1, vector<KeyPoint> ps2, float wholeAreaImg2, int iters = 100) {
    pair<Mat, vector<bool>> homographyWithInliers = ransacContrario<DMatch, Mat>(goodMatches, 4,
            [&ps1, &ps2](vector<DMatch> matches) -> Mat {
                vector<Point2f> srcPoints;
                vector<Point2f> dstPoints;
                for (DMatch match : matches) {
                    srcPoints.push_back(ps1[match.queryIdx].pt);
                    dstPoints.push_back(ps2[match.trainIdx].pt);
                }
                return findMyHomography(srcPoints, dstPoints);
            },
            [&ps1, &ps2](Mat h, DMatch match) -> float {
                Point2f p1 = ps1[match.queryIdx].pt;
                Point2f p2 = ps2[match.trainIdx].pt;
                float p1Data[] = {p1.x, p1.y, 1};
                Mat m(3, 1, CV_32FC1, p1Data);
                Mat p12m = h * m;
                p12m *= 1 / p12m.at<float>(2, 0);
                Point2f p12 = Point2f(p12m.at<float>(0, 0), p12m.at<float>(1, 0));
                float dist = (float) norm(p2 - p12);
                return dist;
            },
            HOMOGRAPHY_TASK,
            wholeAreaImg2,
            iters
    );
    return homographyWithInliers;
}

vector<int> showCircles = {0, 1};
vector<pair<int, int>> showMatches = {pair<int, int>(0, 1)};//pair<int, int>()

vector<vector<vector<DMatch>>> findMatches(vector<vector<KeyPoint>> keyPss, vector<Mat> descrs, float threshold, bool silent);

vector<vector<vector<DMatch>>> filterMatchesByVoting(vector<vector<vector<DMatch>>> matches, vector<vector<KeyPoint>> pss, vector<Mat> imgs, bool silent = true);

void showPerspective(vector<DMatch> matches, vector<KeyPoint> pss1, vector<KeyPoint> pss2, Mat img1, Mat img2, Mat H, int i, int j);

void calculateNodesViaACRansac(vector<vector<vector<DMatch>>> matches, vector<vector<KeyPoint>> keyPss, vector<Mat> imgs, ImgNode nodes[]);

pair<vector<vector<KeyPoint>>, vector<Mat>> loadFromFileOrCalc(const char *cache, vector<Mat> imgs) {
    string imageNumberKey("imageNumber");
    string keyPointsKey("keyPoints");
    string descriptorsKey("descriptors");
    FileStorage fsR(cache, FileStorage::READ);
    pair<vector<vector<KeyPoint>>, vector<Mat>> res;
    int n = (int) imgs.size();
    if (fsR.isOpened()) {
        int fileN;
        fsR[imageNumberKey.data()] >> fileN;
        if (fileN == n) {
            vector<vector<KeyPoint>> keyPs;
            for (int i = 0; i < n; i++) {
                keyPs.push_back(vector<KeyPoint>());
                FileNode node = fsR[(keyPointsKey + to_string(i)).data()];
                read(node, keyPs[i]);
            }
            vector<Mat> descrs;
            for (int i = 0; i < n; i++) {
                descrs.push_back(Mat());
                FileNode node = fsR[(descriptorsKey + to_string(i)).data()];
                read(node, descrs[i]);
            }
            res = pair<vector<vector<KeyPoint>>, vector<Mat>>(keyPs, descrs);
        } else {
            cout << "File cache is outdated!" << endl;
        }
    }
    if (res.first.size() == 0) {
        res = computeDescriptors(imgs, 8000, false);
        FileStorage fsW(cache, FileStorage::WRITE);

        vector<vector<KeyPoint>> keyPs = res.first;
        fsW << imageNumberKey.data() << n;
        for (int i = 0; i < n; i++) {
            write(fsW, keyPointsKey + to_string(i), keyPs[i]);
        }
        vector<Mat> descrs = res.second;
        for (int i = 0; i < n; i++) {
            write(fsW, descriptorsKey + to_string(i), descrs[i]);
        }
    }
    return res;
}

vector<vector<vector<DMatch>>> loadMatchesOrCalc(char const *cache, vector<vector<KeyPoint>> keyPss, vector<Mat> descrs, float threshold = 0.75, bool silent = true) {
    vector<vector<vector<DMatch>>> res;
    string matchKey("match");
    string matchesImageNumberKey("matchesImageNumber");
    int n = (int) descrs.size();

    FileStorage fsR(cache, FileStorage::READ);
    if (fsR.isOpened()) {
        int fileN;
        fsR[matchesImageNumberKey.data()] >> fileN;
        if (fileN == n) {
            for (int i = 0; i < n; i++) {
                res.push_back(vector<vector<DMatch>>());
                for (int j = 0; j < i; j++) {
                    res[i].push_back(vector<DMatch>());
                    FileNode node = fsR[(matchKey + to_string(i) + "-" + to_string(j)).data()];
                    read(node, res[i][j]);
                }
            }
        } else {
            cout << "File cache is outdated!" << endl;
        }
    }
    if (res.size() == 0) {
        res = findMatches(keyPss, descrs, threshold, silent);
        FileStorage fsW(cache, FileStorage::WRITE);
        fsW << matchesImageNumberKey.data() << n;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                write(fsW, matchKey + to_string(i) + "-" + to_string(j), res[i][j]);
            }
        }
    }
    return res;
}

const int minKeyPoints = 8;
const int minMatches = 8;

int main(int argc, char **argv) {
    vector<Mat> imgs = loadAllImages();
    unsigned long n = imgs.size();
    cout << "Images count: " << n << endl;

    pair<vector<vector<KeyPoint>>, vector<Mat>> pointsAndDescrs = loadFromFileOrCalc("cache.dat", imgs);

    vector<vector<KeyPoint>> keyPss = pointsAndDescrs.first;
    vector<Mat> descrs = pointsAndDescrs.second;

    for (int i : showCircles) {
        Mat withCircles = drawCircles(keyPss[i], imgs[i], Scalar(200, 100, 100), 10, 2);
        imshow(to_string(i) + string(" with circles"), scaleToFit(withCircles, 800));
        waitKey(10);
    }

    auto matches = loadMatchesOrCalc("cacheMatches.dat", keyPss, descrs, 1, false);
    matches = filterMatchesByVoting(matches, keyPss, imgs, false);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (matches[i][j].size() > 0) {
                cout << "Images " << i << "-" << j << " matches count: " << matches[i][j].size() << endl;
            }
        }
    }

    ImgNode nodes[n];
    calculateNodesViaACRansac(matches, keyPss, imgs, nodes);
    drawComponents(nodes, n);

    waitKey();
}

void calculateNodesViaACRansac(vector<vector<vector<DMatch>>> matches, vector<vector<KeyPoint>> keyPss, vector<Mat> imgs, ImgNode nodes[]) {
    int n = (int) matches.size();
    for (int i = 0; i < n; i++) {
        nodes[i] = ImgNode(i, Mat(), imgs[i]);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (matches[i][j].size() < minMatches) {
                if (matches[i][j].size() >= 5) {
                    cout << i << "-" << j << " matches count is too low: " << matches[i][j].size() << endl;
                }
                continue;
            }
            if (find(showMatches.begin(), showMatches.end(), pair<int, int>(j, i)) != showMatches.end()) {
                Mat withMatches;
                drawMatches(imgs[i], keyPss[i], imgs[j], keyPss[j], matches[i][j], withMatches);
                imshow(to_string(i) + string("-") + to_string(j) + " matches (count=" + to_string(matches[i][j].size()) + ")",
                        scaleToFit(withMatches, 800));
            }

            pair<Mat, vector<bool>> homographyWithInliers = findHomographyACRansac(matches[i][j], keyPss[i], keyPss[j], imgs[j].rows * imgs[j].cols, 100);
            Mat H = homographyWithInliers.first;
            vector<bool> isInlier = homographyWithInliers.second;
            long outliersCount = count(isInlier.begin(), isInlier.end(), false);
            long inliersCount = count(isInlier.begin(), isInlier.end(), true);
            if (H.empty()) {
                cout << " Too many outliers in " << matches[i][j].size() << " matches " << i << "-" << j << ": outliersCount=" << outliersCount << endl;
                continue;
            }

            pair<Size, Mat> perspectiveSAndMove = getPerspectiveSizeAndMovement(Size(imgs[i].cols, imgs[i].rows), Size(imgs[j].cols, imgs[j].rows), H);
            if (perspectiveSAndMove.first.height * perspectiveSAndMove.first.width > 4 * (imgs[i].cols * imgs[i].rows + imgs[j].cols * imgs[j].rows)) {
                cout << i << "-" << j << " Too large perspective!!! " << perspectiveSAndMove.first << endl;
                continue;
            }
            Mat perspective(perspectiveSAndMove.first, CV_32FC1);
            Mat move = perspectiveSAndMove.second;
            warpPerspective(imgs[j], perspective, move, perspective.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
            warpPerspective(imgs[i], perspective, move * H, perspective.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);
            drawCirclesWithDist(keyPss[i], keyPss[j], matches[i][j], isInlier, perspective, move * H, move);

            cout << i << "-" << j << " inliers/matches: " << inliersCount << "/" << matches[i][j].size() << endl;
            nodes[i].to.push_back(j);
            nodes[j].to.push_back(i);
            nodes[i].revHomo.push_back(H.inv());
            nodes[j].revHomo.push_back(H);
        }
    }
}

float angleBetween(Point v1, Point v2) {
    float dy = v2.y - v1.y;
    float dx = v2.x - v1.x;
    return atan2(dy, dx);
}

Mat voteForAngle(vector<DMatch> matches, vector<KeyPoint> pss1, vector<KeyPoint> pss2, float distanceThreshold, int traceI = -1, int traceJ = -1) {
    int n = (int) matches.size();
    long degreesBuckets = 100;
    int kernel = 7;
    if (n > degreesBuckets) {
        degreesBuckets = 400;
        kernel = 15;
    }
    Mat degreesVotes = Mat::zeros(1, degreesBuckets * 2, CV_32FC1);
    int thresholdedCount = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                continue;
            }
            DMatch m1 = matches[i];
            DMatch m2 = matches[j];
            Point2f p11 = pss1[m1.queryIdx].pt;
            Point2f p12 = pss1[m2.queryIdx].pt;
            Point2f p21 = pss2[m1.trainIdx].pt;
            Point2f p22 = pss2[m2.trainIdx].pt;
            if (abs(norm(p12 - p11) - norm(p22 - p21)) > distanceThreshold) {
                thresholdedCount++;
                continue;
            }
            float a1 = angleBetween(p11, p12);
            float a2 = angleBetween(p21, p22);
            float angle = (float) ((a2 - a1) + M_PI);
            if (angle > 2 * M_PI) {
                angle -= 2 * M_PI;
            }
            if (angle < 0) {
                angle += 2 * M_PI;
            }
            int id = (int) (angle * degreesBuckets / (2 * M_PI));
            assert (id >= 0 && id < degreesBuckets);
            degreesVotes.at<float>(0, id) += 1;
            degreesVotes.at<float>(0, id + degreesBuckets) += 1;
        }
    }

    {
        drawPlot("DiagramG", degreesVotes, degreesBuckets * 4);
        Mat gaused;
        cv::GaussianBlur(degreesVotes, gaused, cv::Size(kernel, 1), 0, 0);
        degreesVotes = Mat(1, degreesBuckets, CV_32FC1);
        for (int c = 0; c < degreesBuckets; ++c) {
            degreesVotes.at<float>(0, c) = gaused.at<float>(0, c + degreesBuckets / 2);
        }
        drawPlot("Diagram", degreesVotes, degreesBuckets * 2, 700, true);
    }

    int minI = 0;
    for (int i = 1; i < degreesBuckets; ++i) {
        if (degreesVotes.at<float>(0, i) < degreesVotes.at<float>(0, minI)) {
            minI = i;
        }
    }
    int biggest[2] = {-1, -1};
    for (int cur = 0; cur < 2; ++cur) {
        for (int i = 0; i < degreesBuckets; i++) {
            if (i == biggest[0]) {
                continue;
            }
            if ((biggest[cur] == -1 || degreesVotes.at<float>(0, i) > degreesVotes.at<float>(0, biggest[cur]))
                    && degreesVotes.at<float>(0, (i - 1 + degreesBuckets) % degreesBuckets) < degreesVotes.at<float>(0, i)
                    && degreesVotes.at<float>(0, (i + 1) % degreesBuckets) < degreesVotes.at<float>(0, i)) {
                biggest[cur] = i;
            }
        }
    }
    int biggestB = biggest[0];
    int secondBiggestB = biggest[1];
    int minV = degreesVotes.at<float>(0, minI);
    cout << degreesVotes << endl;
    if (degreesVotes.at<float>(0, biggestB) - minV <= (degreesVotes.at<float>(0, secondBiggestB) - minV) * 1.5 || degreesVotes.at<float>(0, biggestB) <= 3) {
        if (traceI != -1 && traceJ != -1) {
            cout << "Images " << traceI << "-" << traceJ << " Voting for angle failed! Biggest: " << degreesVotes.at<float>(0, biggestB) << ", " << degreesVotes.at<float>(0, secondBiggestB)
                    << " (at: " << biggestB * 2 * M_PI / degreesBuckets - M_PI << ", " << secondBiggestB * 2 * M_PI / degreesBuckets - M_PI << ")" << endl;
        }
        return Mat();
    } else {
        float angle = (float) ((biggestB * 2 * M_PI) / degreesBuckets - M_PI);
        float data[] = {cos(angle), -sin(angle), 0,
                sin(angle), cos(angle), 0,
                0, 0, 1};
        Mat rotation(3, 3, CV_32FC1, data);
        return rotation.clone();
    }
}

Mat voteForShift(vector<DMatch> matches, vector<KeyPoint> pss1, vector<KeyPoint> pss2, Mat img1, Mat img2, int traceI = -1, int traceJ = -1) {
    int n = (int) matches.size();
    const long xBuckets = max(img1.cols / 4, n / 4);
    const long yBuckets = max(img1.rows / 4, n / 4);
    Mat shiftVotes = Mat::zeros(yBuckets + 2, xBuckets + 2, CV_32FC1);
    for (DMatch match : matches) {
        Point2f p1 = pss1[match.queryIdx].pt;
        Point2f p2 = pss2[match.trainIdx].pt;
        Point2f dp = p2 - p1;
        if (abs(dp.x) > img1.cols || abs(dp.y) > img1.rows) {
            continue;
        }
        int idX = (dp.x + img1.cols) * xBuckets / (2 * img1.cols);
        int idY = (dp.y + img1.rows) * yBuckets / (2 * img1.rows);
        assert (idX >= 0 && idX < xBuckets);
        assert (idY >= 0 && idY < yBuckets);
        shiftVotes.at<float>(idY + 1, idX + 1) += 1;
    }
    Mat gaussed;
    int kernel = 65;
    cv::GaussianBlur(shiftVotes, gaussed, cv::Size(kernel, kernel), 0.3 * (kernel / 2 - 1) + 0.8, 0.3 * (kernel / 2 - 1) + 0.8);
    shiftVotes = gaussed;
    int biggestX = 0;
    int biggestY = 0;
    int secondBiggestX = 0;
    int secondBiggestY = 0;
    for (int x = 1; x <= xBuckets; x++) {
        for (int y = 1; y <= yBuckets; y++) {
            if (shiftVotes.at<float>(y, x) > shiftVotes.at<float>(biggestY, biggestX)) {
                bool isLocalMaximum = true;
                int dx[] = {-1, 0, 1, 1, 1, 0, -1, -1};
                int dy[] = {1, 1, 1, 0, -1, -1, -1, 0};
                for (int i = 0; i < 8; i++) {
                    if (shiftVotes.at<float>(y + dy[i], x + dx[i]) > shiftVotes.at<float>(y, x)) {
                        isLocalMaximum = false;
                        break;
                    }
                }
                if (isLocalMaximum) {
                    secondBiggestX = biggestX;
                    secondBiggestY = biggestY;
                    biggestX = x;
                    biggestY = y;
                }
            }
        }
    }
    if (shiftVotes.at<float>(biggestY, biggestX) <= shiftVotes.at<float>(secondBiggestY, secondBiggestX) * 1.5) {
        if (traceI != -1 && traceJ != -1) {
            cout << "Voting for shift failed! Biggest: " << shiftVotes.at<float>(biggestY, biggestX) << ", " << shiftVotes.at<float>(secondBiggestY, secondBiggestX)
                    << ". i=" << traceI << ", j=" << traceJ << endl;
        }
        return Mat();
    } else {
        float shiftX = biggestX * (2 * img1.cols) / xBuckets - img1.cols;
        float shiftY = biggestY * (2 * img1.rows) / yBuckets - img1.rows;
        float data[] = {1, 0, shiftX,
                0, 1, shiftY,
                0, 0, 1};
        return Mat(3, 3, CV_32FC1, data).clone();
    }
}

vector<DMatch> filterMatchesByVoting(vector<DMatch> matches, vector<KeyPoint> pss1, vector<KeyPoint> pss2, Mat img1, Mat img2, int traceI = -1, int traceJ = -1) {

    if (traceI == 15 && traceJ == 12) {
        int x = 239;
    }
    Mat rotation = voteForAngle(matches, pss1, pss2, max(img1.cols, img1.rows) * 0.05f);//, traceI, traceJ);
    if (rotation.empty()) {
        if (traceI != -1 && traceJ != -1) {
            cout << "Images " << traceI << "-" << traceJ << " filtered matches result: " << matches.size() << "->" << "no rotation found" << endl;
        }
        return vector<DMatch>();
    }
    vector<KeyPoint> pss1Rotated;
    for (KeyPoint p : pss1) {
        pss1Rotated.push_back(p);
    }
    for (DMatch m:matches) {
        Point2f p = pss1[m.queryIdx].pt;
        float data[] = {p.x, p.y, 1};
        Mat pMat(3, 1, CV_32FC1, data);
        Mat pRotMat = rotation * pMat;
        pss1Rotated[m.queryIdx].pt = Point2f(pRotMat.at<float>(0, 0), pRotMat.at<float>(1, 0));
    }

    Mat shift = voteForShift(matches, pss1Rotated, pss2, img1, img2);//, traceI, traceJ);
    if (shift.empty()) {
        if (traceI != -1 && traceJ != -1) {
            cout << "Images " << traceI << "-" << traceJ << " filtered matches result: " << matches.size() << "->" << "no shift found" << endl;
        }
        return vector<DMatch>();
    }

    Mat H = shift * rotation;
    vector<DMatch> filtered;

//    cout << traceI << "-" << traceJ << endl;
//    showPerspective(matches, pss1, pss2, img1, img2, H, traceI, traceJ);//DEBUG
    for (DMatch m:matches) {
        Point2f p1 = pss1[m.queryIdx].pt;
        float data[] = {
                p1.x,
                p1.y,
                1,
        };
        Mat p1Mat(3, 1, CV_32FC1, data);
        Mat p1Im = H * p1Mat;
        p1Im /= p1Im.at<float>(2, 0);
        float x = p1Im.at<float>(0, 0);
        float y = p1Im.at<float>(0, 1);

        float outCoordinatesThreshold = 0.05;
        if (x > -img2.cols * outCoordinatesThreshold && x < img2.cols * (1 + outCoordinatesThreshold)
                && y > -img2.rows * outCoordinatesThreshold && y < img2.rows * (1 + outCoordinatesThreshold)) {
            filtered.push_back(m);
        }
    }

    if (traceI != -1 && traceJ != -1) {
        cout << "Images " << traceI << "-" << traceJ << " filtered matches result success: " << matches.size() << "->" << filtered.size() << endl;
    }
    return filtered;
}

void showPerspective(vector<DMatch> matches, vector<KeyPoint> pss1, vector<KeyPoint> pss2, Mat img1, Mat img2, Mat H, int i, int j) {
    pair<Size, Mat> perspectiveSAndMove = getPerspectiveSizeAndMovement(Size(img1.cols, img1.rows), Size(img1.cols, img1.rows), H);
    cout << perspectiveSAndMove.first << endl;
    if (perspectiveSAndMove.first.height * perspectiveSAndMove.first.width > 10000 * 10000) {
        cout << "Too large perspective!!! " << perspectiveSAndMove.first << endl;
        return;
    }
    vector<KeyPoint> tmp1;
    vector<KeyPoint> tmp2;
    for (DMatch match : matches) {
        tmp1.push_back(pss1[match.queryIdx]);
        tmp2.push_back(pss2[match.trainIdx]);
    }
    img1 = drawCircles(tmp1, img1, Scalar(0, 0, 255), 10, 5);
    img2 = drawCircles(tmp2, img2, Scalar(150, 0, 150), 10, 5);

    Mat perspective12(perspectiveSAndMove.first, CV_32FC1);
    Mat move = perspectiveSAndMove.second;
    warpPerspective(img2, perspective12, move, perspective12.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    warpPerspective(img1, perspective12, move * H, perspective12.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);

    Mat perspective21(perspectiveSAndMove.first, CV_32FC1);
    warpPerspective(img1, perspective21, move * H, perspective21.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    warpPerspective(img2, perspective21, move, perspective21.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);

    int frame = 0;
    Mat imgs[] = {perspective12, perspective21};
    while (waitKey(200) == -1) {
        imshow(string("DebugShow ") + to_string(i) + "-" + to_string(j), scaleToFit(imgs[frame]));
        frame = (frame + 1) % 2;
    }
}

vector<vector<vector<DMatch>>> filterMatchesByVoting(vector<vector<vector<DMatch>>> matches, vector<vector<KeyPoint>> pss, vector<Mat> imgs, bool silent) {
    int n = matches.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (matches[i][j].size() == 0) {
                continue;
            }
            if (silent) {
                matches[i][j] = filterMatchesByVoting(matches[i][j], pss[i], pss[j], imgs[i], imgs[j]);
            } else {
                matches[i][j] = filterMatchesByVoting(matches[i][j], pss[i], pss[j], imgs[i], imgs[j], i, j);
            }
        }
    }
    return matches;
}

vector<vector<vector<DMatch>>> findMatches(vector<vector<KeyPoint>> keyPss, vector<Mat> descrs, float threshold = 0.75, bool silent = true) {
    long n = keyPss.size();
    vector<vector<vector<DMatch>>> matches(n);
    for (int i = 0; i < n; i++) {
        matches[i] = vector<vector<DMatch>>(n);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (keyPss[i].size() >= minKeyPoints && keyPss[j].size() >= minKeyPoints) {
                vector<DMatch> cur;
                if (silent) {
                    cur = findMatches(descrs[i], descrs[j], threshold);
                } else {
                    cur = findMatches(descrs[i], descrs[j], threshold, i, j);
                }
                if (cur.size() >= minMatches) {
                    matches[i][j] = cur;
                }
            }
        }
    }
    return matches;
}


void calcRadius(int n, ImgNode nodes[], int dist[]) {
    int q[n];
    bool used[n];
    for (int i = 0; i < n; i++) {
        used[i] = false;
    }
    int next = 0;
    int last = -1;
    for (int i = 0; i < n; i++) {
        ImgNode node = nodes[i];
        if (node.to.size() <= 1) {
            q[last++] = node.id;
            dist[node.id] = 0;
            used[node.id] = true;
        }
    }
    while (next < last) {
        int i = q[next++];
        for (int j : nodes[i].to) {
            if (!used[j]) {
                used[j] = true;
                q[last++] = j;
                dist[j] = dist[i] + 1;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        if (!used[i]) {
            dist[i] = -1;
        }
    }
}

vector<int> drawComponent(ImgNode nodes[], bool used[], int cur, int prev, Mat prevH, int root) {
    used[cur] = true;
    Mat oldComponent = nodes[root].componentImage;
//    imshow(to_string(prev) + " <- " + to_string(cur), scaleToFit(oldComponent));
//    waitKey();

    Mat H = nodes[prev].toComponentTransformation * prevH;
    pair<Size, Mat> perspectiveSAndMove = getPerspectiveSizeAndMovement(
            Size(nodes[cur].img.cols, nodes[cur].img.rows),
            Size(nodes[root].componentImage.cols, nodes[root].componentImage.rows),
            H);
    nodes[root].componentImage = Mat(perspectiveSAndMove.first, CV_32FC1);
    Mat move = perspectiveSAndMove.second;
    warpPerspective(oldComponent, nodes[root].componentImage, move, nodes[root].componentImage.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    warpPerspective(nodes[cur].img, nodes[root].componentImage, move * H, nodes[root].componentImage.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);
    nodes[cur].toComponentTransformation = move * H;
    nodes[root].transformations.push_back(move);
//    imshow(to_string(prev) + " <- " + to_string(cur), scaleToFit(nodes[root].componentImage));
//    waitKey();

    vector<int> subComponentIndexes = {cur};
    long wasRootTransormations = nodes[root].transformations.size();
    for (int i = 0; i < nodes[cur].to.size(); i++) {
        int to = nodes[cur].to[i];
        Mat revH = nodes[cur].revHomo[i];
        if (!used[to]) {
            vector<int> v = drawComponent(nodes, used, to, cur, revH, root);
            for (int vi : v) {
                subComponentIndexes.push_back(vi);
            }
        }
        for (int i = wasRootTransormations; i < nodes[root].transformations.size(); i++) {
            nodes[cur].toComponentTransformation = nodes[root].transformations[i] * nodes[cur].toComponentTransformation;
        }
        wasRootTransormations = nodes[root].transformations.size();
    }
    return subComponentIndexes;
}

void drawComponent(ImgNode nodes[], bool used[], int cur) {
    used[cur] = true;
    nodes[cur].componentImage = nodes[cur].img;
    nodes[cur].toComponentTransformation = Mat::eye(3, 3, CV_32FC1);
    vector<int> imgs = {cur};
    long wasRootTransormations = nodes[cur].transformations.size();
    for (int i = 0; i < nodes[cur].to.size(); i++) {
        int to = nodes[cur].to[i];
        Mat revH = nodes[cur].revHomo[i];
        if (!used[to]) {
            vector<int> v = drawComponent(nodes, used, to, cur, revH, cur);
            for (int vi : v) {
                imgs.push_back(vi);
            }
        }
        for (int i = wasRootTransormations; i < nodes[cur].transformations.size(); i++) {
            nodes[cur].toComponentTransformation = nodes[cur].transformations[i] * nodes[cur].toComponentTransformation;
        }
        wasRootTransormations = nodes[cur].transformations.size();
    }
    string indexes = to_string(cur);
    for (int i = 1; i < imgs.size(); i++) {
        indexes += ", " + to_string(imgs[i]);
    }
    if (imgs.size() > 1) {
        imshow(indexes, scaleToFit(nodes[cur].componentImage, 800));
    } else {
        imshow(indexes, scaleToFit(nodes[cur].componentImage, 300));
    }
}

void drawComponents(ImgNode nodes[], int n) {
    bool used[n];
    for (int i = 0; i < n; i++) {
        used[i] = false;
    }
    int dist[n];
    calcRadius(n, nodes, dist);
    for (int iter = 0; iter < n; iter++) {
        int maxI = -1;
        for (int i = 0; i < n; i++) {
            if (!used[i] && (maxI == -1 || (dist[i] > dist[maxI] || (dist[i] == dist[maxI] && nodes[i].to.size() >= nodes[maxI].to.size())))) {
                maxI = i;
            }
        }
        if (maxI == -1) {
            break;
        }
        drawComponent(nodes, used, maxI);
    }
}