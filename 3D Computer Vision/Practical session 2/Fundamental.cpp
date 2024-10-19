// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

// Student: Lucas Versini

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// Computes the fundamental matrix based on 8 matches.
FMatrix<float, 3, 3> algorithm8points(vector<Match>& matches, vector<int> points){
    const float normalization = 0.001;
    Matrix<float> A(max(9, (int) points.size()), 9);
    float x1, y1, x2, y2;

    // Fill A (with normalization)
    for (int i = 0; i < points.size(); i++){
        x1 = normalization * matches[points[i]].x1;
        y1 = normalization * matches[points[i]].y1;
        x2 = normalization * matches[points[i]].x2;
        y2 = normalization * matches[points[i]].y2;

        A(i, 0) = x1 * x2;
        A(i, 1) = x1 * y2;
        A(i, 2) = x1;

        A(i, 3) = y1 * x2;
        A(i, 4) = y1 * y2;
        A(i, 5) = y1;

        A(i, 6) = x2;
        A(i, 7) = y2;
        A(i, 8) = 1;
    }

    // Add the constraint 0^T f = 0 if necessary
    if (points.size() < 9){
        for (int i = 0; i < 9; i++) A(8, i) = 0;
    }

    // Perform SVD, and fill F
    Matrix<float> U, V; Vector<float> S;
    svd(A, U, S, V);

    FMatrix<float, 3, 3> F;
    int row = V.nrow() - 1;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            F(i, j) = V(row, 3 * i + j);
        }
    }

    // Make F rank 2.
    FMatrix<float, 3, 3> U2, V2; FVector<float, 3> S2;
    svd(F, U2, S2, V2);
    S2[2] = 0;
    F = U2 * Diagonal(S2) * V2;

    // De-normalize.
    FMatrix<float, 3, 3> N(0.);
    N(0, 0) = N(1, 1) = normalization; N(2, 2) = 1;

    return transpose(N) * F * N; // Note that the transpose is not useful for this N.
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    // --------------- DONE ------------
    // WITH NORMALIZATION OF POINTS
    for (int i = 0; i < Niter; i++){
        // Randomly select matches
        vector<int> inliers, points;
        for (int j = 0; j < 8; j++) points.push_back(rand() % matches.size());

        // Apply 8-point algorithm
        FMatrix<float, 3, 3> F = algorithm8points(matches, points);

        // Find inliers
        for (int j = 0; j < matches.size(); j++){
            FVector<float, 3> x(matches[j].x1, matches[j].y1, 1);
            x = transpose(F) * x;
            FVector<float, 3> xprime(matches[j].x2, matches[j].y2, 1);

            float dist = abs(x * xprime) / sqrt(x[0] * x[0] + x[1] * x[1]);
            if (dist < distMax) inliers.push_back(j);
        }

        // Keep F and the inliers if this is the best result yet
        if (inliers.size() > bestInliers.size()){
            bestInliers = inliers;
            float den = log(1 - pow(((float) inliers.size()) / matches.size(), 8));
            if (den < 0) Niter = std::min(Niter, ((int) ceil(log(BETA) / den))); // To avoid log(1 - x) = 0 for small x
        }
    }
    // Recompute F from the best inliers only
    bestF = algorithm8points(matches, bestInliers);

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);
    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;
        // --------------- DONE ------------

        fillCircle(x, y, 5, BLUE);

        bool leftImage = (x < I1.width());
        FVector<float, 3> p(x, y, 1.);

        if (leftImage){
            // Right epipolar line l = F^T p, with equation l[0] x + l[1] y + l[2] = 0
            p = transpose(F) * p;
            float yLeft = -p[2] / p[1];
            float yRight = -(p[0] * I1.width() + p[2]) / p[1];
            drawLine(I1.width(), yLeft, I1.width() + I2.width(), yRight, RED);
        }
        else {
            // Left epipolar line l = F p', with equation l[0] x + l[1] y + l[2] = 0
            p[0] -= I1.width();
            p = F * p;
            float yLeft = -p[2] / p[1];
            float yRight = -(p[0] * I1.width() + p[2]) / p[1];
            drawLine(0, yLeft, I1.width(), yRight, RED);
        }
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
