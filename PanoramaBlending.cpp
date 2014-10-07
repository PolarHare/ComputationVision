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

const int minMatches = 8;
const int middleMatches = 30;
const int orbPointsDefault = 1500;
const float maxOutliers = 0.25f;

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

Mat drawCircles(vector<KeyPoint> ps, Mat img, Scalar color = Scalar(200, 100, 100), int radius = 4, int tickness = 1) {
    Mat res;
    img.copyTo(res);
    for (KeyPoint p : ps) {
        circle(res, p.pt, radius, color, tickness, LINE_8);
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

pair<vector<vector<KeyPoint>>, vector<Mat>> computeDescriptors(vector<Mat> imgsForPs, vector<Mat> imgsForDescrs, int orbPoints = orbPointsDefault, bool silent = true) {
    ORB orb(orbPoints);
    vector<vector<KeyPoint>> keyPss;
    for (int i = 0; i < imgsForPs.size(); i++) {
        keyPss.push_back(vector<KeyPoint>());
        orb.detect(imgsForPs[i], keyPss[i]);
        if (!silent) {
            cout << "Image" << i << " key points count: " << keyPss[i].size() << endl;
        }
    }

    vector<Mat> descrs;
    for (int i = 0; i < imgsForDescrs.size(); i++) {
        descrs.push_back(Mat());
        orb.compute(imgsForDescrs[i], keyPss[i], descrs[i]);
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
        if (match12[0].distance < match12[1].distance * 0.6) {
            goodMatches.push_back(match12[0]);
        }
    }
    cout << "Good matches/all matches for i=" << i << " and j=" << j << ": " << goodMatches.size() << "/" << matches.size() << endl;
    return goodMatches;
}

pair<Mat, vector<bool>> findHomographyACRansac(vector<DMatch> goodMatches, vector<KeyPoint> ps1, vector<KeyPoint> ps2, float wholeAreaImg2) {
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
            wholeAreaImg2
    );
    return homographyWithInliers;
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

void drawComponents(ImgNode vector[], int n);

void drawComponent(ImgNode vector[], bool pBoolean[], int i);

void calcRadius(int n, ImgNode nodes[], int pInt[]);

vector<DMatch> findOverlapingMatchesByVoting(vector<DMatch> matches, vector<KeyPoint> keyPss1, vector<KeyPoint> keyPss2, Size2i size1, Size2i size2, int traceI = -1, int traceJ = -1);

int main(int argc, char **argv) {
    vector<Mat> imgs = loadAllImages();
    vector<Mat> imgsGray = loadAllImages(IMREAD_GRAYSCALE);
    unsigned long n = imgsGray.size();

    for (int i = 0; i < n; i++) {
        createCLAHE()->apply(imgsGray[i], imgsGray[i]);
    }

    pair<vector<vector<KeyPoint>>, vector<Mat>> pointsAndDescrs = computeDescriptors(imgs, imgs, orbPointsDefault, false);
    vector<vector<KeyPoint>> keyPss = pointsAndDescrs.first;
    vector<Mat> descrs = pointsAndDescrs.second;

    for (int i = 0; i < n; i++) {
        Mat withCircles = drawCircles(keyPss[i], imgs[i], Scalar(200, 100, 100), 10, 2);
        imshow(to_string(i) + string(" with circles"), scaleToFit(withCircles, 400));
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

    ImgNode nodes[n];
    vector<int> showMatches = {14, 7};
    vector<int> showPers = {};
    for (int i = 0; i < n; i++) {
//        Mat withCircles = drawCircles(keyPss[i], imgs[i], Scalar(200, 100, 100), 10);
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
            if (matches[i][j].size() < middleMatches) {
                pair<vector<vector<KeyPoint>>, vector<Mat>> betterPointsAndDescrs =
                        computeDescriptors(vector<Mat>{imgs[i], imgs[j]}, vector<Mat>{imgs[i], imgs[j]}, orbPointsDefault * 2, true);
                cout << " Increased points count (" << i << "-" << j << "): " << "i.keyPoints=" << betterPointsAndDescrs.first[0].size() << ", j.keyPoints=" << betterPointsAndDescrs.first[1].size() << endl;
                keyPss[i] = betterPointsAndDescrs.first[0];
                keyPss[j] = betterPointsAndDescrs.first[1];
                descrs[i] = betterPointsAndDescrs.second[0];
                descrs[j] = betterPointsAndDescrs.second[1];
                long oldMatches = matches[i][j].size();
                matches[i][j] = findMatches(descrs[i], descrs[j], i, j);
                matches[j][i] = matches[i][j];
                cout << " " << i << "-" << j << " matches count changed: " << oldMatches << "->" << matches[i][j].size() << endl;
                if (matches[i][j].size() < 2.0f * oldMatches && matches[i][j].size() < middleMatches) {
                    cout << "Didn't help!!!" << endl;
                    continue;
                }
            }
            if (find(showMatches.begin(), showMatches.end(), i) != showMatches.end()
                    && find(showMatches.begin(), showMatches.end(), j) != showMatches.end()) {
                Mat withMatches;
                drawMatches(imgs[i], keyPss[i], imgs[j], keyPss[j], matches[i][j], withMatches);
                imshow(to_string(i) + string("-") + to_string(j) + " matches (count=" + to_string(matches[i][j].size()) + ")",
                        scaleToFit(withMatches, 800));
            }

            vector<DMatch> matchesByVoting = findOverlapingMatchesByVoting(matches[i][j], keyPss[i], keyPss[j],
                    Size(imgs[i].cols, imgs[i].rows), Size(imgs[j].cols, imgs[j].rows), i, j);

            pair<Mat, vector<bool>> homographyWithInliers = findHomographyACRansac(matches[i][j], keyPss[i], keyPss[j], imgs[j].rows * imgs[j].cols);
            Mat H = homographyWithInliers.first;
            vector<bool> isInlier = homographyWithInliers.second;
            long outliersCount = count(isInlier.begin(), isInlier.end(), false);
            long inliersCount = count(isInlier.begin(), isInlier.end(), true);
            if (H.empty()) {
                cout << " Too many outliers in " << matches[i][j].size() << " matches " << i << "-" << j << ": outliersCount=" << outliersCount << endl;
                continue;
            }

            pair<Size, Mat> perspectiveSAndMove = getPerspectiveSizeAndMovement(Size(imgs[i].cols, imgs[i].rows), Size(imgs[j].cols, imgs[j].rows), H);
            cout << perspectiveSAndMove.first << endl;
            if (perspectiveSAndMove.first.height * perspectiveSAndMove.first.width > 10000 * 10000) {
                cout << "Too large perspective!!! " << perspectiveSAndMove.first << endl;
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

    drawComponents(nodes, n);
    waitKey();
}

float angleBetween(Point v1, Point v2) {
    float dy = v2.y - v1.y;
    float dx = v2.x - v1.x;
    return atan2(dy, dx);
}

vector<DMatch> findOverlapingMatchesByVoting(vector<DMatch> matches, vector<KeyPoint> keyPss1, vector<KeyPoint> keyPss2, Size2i size1, Size2i size2, int traceI, int traceJ) {
    long n = matches.size();
    const int degreesBuckets = 360;
    int degreesVotes[degreesBuckets];
    for (int i = 0; i < degreesBuckets; i++) {
        degreesVotes[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            DMatch m1 = matches[i];
            DMatch m2 = matches[j];
            float a1 = angleBetween(keyPss1[m1.queryIdx].pt, keyPss1[m2.queryIdx].pt);
            float a2 = angleBetween(keyPss2[m1.trainIdx].pt, keyPss2[m2.trainIdx].pt);
            float angle = (a2 - a1) + PI;
            if (angle > 2 * PI) {
                angle -= 2 * PI;
            }
            if (angle < 0) {
                angle += 2 * PI;
            }
            int id = (int) (angle * degreesBuckets / (2 * PI));
            assert (id >= 0 && id < degreesBuckets);
            degreesVotes[id]++;
        }
    }
    int biggestB = -1;
    int secondBiggestB = -1;
    for (int i = 0; i < degreesBuckets; i++) {
        if (biggestB == -1 ||
                (degreesVotes[i] > degreesVotes[biggestB]
                        && degreesVotes[i - 1] < degreesVotes[i]
                        && degreesVotes[(i + 1) % degreesBuckets] < degreesVotes[i])) {
            secondBiggestB = biggestB;
            biggestB = i;
        }
    }
    if (degreesVotes[biggestB] <= degreesVotes[secondBiggestB] * 1.5) {
        cout << "Voting for angle failed! Biggest: " << degreesVotes[biggestB] << ", " << degreesVotes[secondBiggestB]
                << ". i=" << traceI << ", j=" << traceJ << endl;
        return vector<DMatch>();
    }
//    vector<float> values(degreesBuckets);
//    for (int i = 0; i < degreesBuckets; i++) {
//        values.push_back(degreesVotes[i]);
//    }
//    printPlot((to_string(traceI) + "-" + to_string(traceJ) + " angle gistogram").data(), values);
    return vector<DMatch>();
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
