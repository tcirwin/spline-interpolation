#include <iostream>
#include "process_curve.h"
#include "init_curves.h"
#include "spline.h"

using namespace std;

// Resolution: the number of discrete line segments in each curve.
//  numPoints: the number of points to plot; number of splines is
//             thus numPoints - 2.
//    numSets: the number of spline curve sets to calculate.
int resolution, numPoints, numSets;

Point** finVals;
Point** sets;

void Splines::init() {
   cin >> resolution >> numPoints >> numSets;

   sets = new Point*[numSets];

   for (int set = 0; set < numSets; set++) {
      sets[set] = new Point[numPoints];

      for (int point = 0; point < numPoints; point++) {
         cin >> sets[set][point].x >> sets[set][point].y;
      }
   }

   finVals = generatePoints(sets, numSets, numPoints, resolution);
}

void Splines::iterate(
 void (*process)(int setNum, int numPoints, int resolution, Point* vals)) {
   for (int set = 0; set < numSets; set++) {
      process(set, numPoints, resolution, finVals[set]);
   }
}

void Splines::cleanup() {
   for (int set = 0; set < numSets; set++) {
      delete [] sets[set];
   }

   delete []sets;
}
