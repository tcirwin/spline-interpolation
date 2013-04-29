#include <iostream>
#include "process_curve.h"
#include "spline.h"

using namespace std;

void processCurve(int set, int numPoints, int resolution, Point* vals) {
   cout << "set " << set << endl;

   for (int point = 0; point < (numPoints - 2) * resolution; point++) {
      cout << vals[point].x << " " << vals[point].y << endl;
   }
}
