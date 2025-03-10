// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Function to display a point
void displayClick(IntPoint2 point, int num) {
    fillCircle(point, 2, YELLOW);
    drawString(IntPoint2(5, 0) + point, to_string(num), YELLOW);
}

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    // ------------- DONE/Complete ----------

    IntPoint2 point;
    Window win;
    int subWin, click;
    int numPoints1 = 0; int numPoints2 = 0;

    // Print instructions for the user
    Window instructions = openWindow(850, 100, "Instructions", 0, 0);
    setActiveWindow(instructions);
    drawString(0, 20,"Click 4 or more corresponding points in both images.", BLACK);
    drawString(0, 50,"Click right to end.", BLACK);
    drawString(0, 80,"Number of points so far:", BLACK);
    drawString(280, 80, "0 / 0", BLACK);

    while (true){
        click = anyGetMouse(point, win, subWin);
        if (click == 3) break; // Right click

        if (win != instructions){
            // Get the click, retrieve the point
            vector<IntPoint2>& pts = (win == w1) ? pts1: pts2;
            if (win == w1) numPoints1++;
            else numPoints2++;;

            pts.push_back(point);
            setActiveWindow(win);

            // Display the click
            displayClick(point, pts.size());

            // Update the instructions
            setActiveWindow(instructions);
            fillRect(280, 60, 200, 80, WHITE);
            drawString(280, 80, to_string(numPoints1)+ " / " + to_string(numPoints2), BLACK);
        }
    }
    cout << "End of points selection." << endl;
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);
    // ------------- DONE/Complete ----------

    for (int i = 0; i < n; i++){
        double x1 = pts1[i].x();
        double y1 = pts1[i].y();
        double x2 = pts2[i].x();
        double y2 = pts2[i].y();

        // Fill A
        A(2 * i, 0) = x1;
        A(2 * i + 1, 0) = 0;
        A(2 * i, 1) = y1;
        A(2 * i + 1, 1) = 0;
        A(2 * i, 2) = 1;
        A(2 * i + 1, 2) = 0;
        A(2 * i, 3) = 0;
        A(2 * i + 1, 3) = x1;
        A(2 * i, 4) = 0;
        A(2 * i + 1, 4) = y1;
        A(2 * i, 5) = 0;
        A(2 * i + 1, 5) = 1;
        A(2 * i, 6) = -x1 * x2;
        A(2 * i + 1, 6) = -x1 * y2;
        A(2 * i, 7) = -y1 * x2;
        A(2 * i + 1, 7) = -y1 * y2;

        // Fill B
        B[2 * i] = x2; B[2 * i + 1] = y2;
    }

    // Solve the system and fill H
    Vector<double> h = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=h[0]; H(0,1)=h[1]; H(0,2)=h[2];
    H(1,0)=h[3]; H(1,1)=h[4]; H(1,2)=h[5];
    H(2,0)=h[6]; H(2,1)=h[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;    
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);
    // ------------- DONE/Complete ----------
    Matrix<float> H_inv = inverse(H);
    Vector<float> v2(3);
    Color color1, color2;

    for (int i = 0; i < I.width(); i++){
        for (int j = 0; j < I.height(); j++) {
            v[0] = i + x0; v[1] = j + y0; v[2] = 1;
            v2 = H_inv * v; v2 /= v2[2];

            // Color of the pixel in image 1
            bool v_in_1 = (0 <= v2[0] && v2[0] < I1.width()
                           && 0 <= v2[1] && v2[1] < I1.height());
            if (v_in_1) color1 = I1.interpolate(v2[0], v2[1]);

            // Color of the pixel in image 2
            bool v_in_2 = (0 <= v[0] && v[0] < I2.width()
                            && 0 <= v[1] && v[1] < I2.height());
            if (v_in_2) color2 = I2.interpolate(v[0], v[1]);

            // Find final color
            if (v_in_1 && v_in_2){
                I(i, j).r() = (color1.r() + color2.r()) / 2;
                I(i, j).g() = (color1.g() + color2.g()) / 2;
                I(i, j).b() = (color1.b() + color2.b()) / 2;
            }
            else if (v_in_2) I(i, j) = color2;
            else if (v_in_1) I(i, j) = color1;
        }
    }

    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
