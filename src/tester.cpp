#include <iostream>
#include <string>
#include "spline.h"

using namespace std;

int main(int argc, char* argv[]) {
   // Resolution: the number of discrete line segments in each curve.
   //  numPoints: the number of points to plot; number of splines is
   //             thus numPoints - 2.
   //    numSets: the number of spline curve sets to calculate.
   int resolution, numPoints, numSets;

   cin >> resolution >> numPoints >> numSets;

   Point** sets = new Point*[numSets];
   Point** finVals;

   for (int set = 0; set < numSets; set++) {
      sets[set] = new Point[numPoints];

      for (int point = 0; point < numPoints; point++) {
         cin >> sets[set][point].x >> sets[set][point].y;
      }
   }

   finVals = generatePoints(sets, numSets, numPoints, resolution);

   for (int set = 0; set < numSets; set++) {
      cout << "set " << set << endl;

      for (int point = 0; point < (numPoints - 2) * resolution; point++) {
         cout << finVals[set][point].x << " " << finVals[set][point].y << endl;
      }

      delete [] sets[set];
      cout << endl;
   }

   delete []sets;
}
