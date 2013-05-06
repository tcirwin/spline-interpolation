#include <iostream>
#include <time.h>
#include <sys/time.h>
#include "process_curve.h"
#include "init_curves.h"
#include "spline.h"
#include "timing.h"

#define BORDER 100

using namespace std;

// Resolution: the number of discrete line segments in each curve.
//  numPoints: the number of points to plot; number of splines is
//             thus numPoints - 2.
//    numSets: the number of spline curve sets to calculate.
int resolution, numPoints, numSets;
// Keep track of largest/smallest x/y vals
float hix = -10000.0, lox = 10000.0, hiy = -10000.0, loy = 10000.0;
float normx = 1, normy = 1;

Point** finVals;
Point* setLin;
Point** sets;

void Splines::init() {
   cout << "Enter [resolution numPoints numSets] followed by " << endl;
   cout << "data points, one per line: ";
   cin >> resolution >> numPoints >> numSets;

   setLin = new Point[numSets * numPoints];
   sets = new Point*[numSets];

   for (int set = 0; set < numSets; set++) {
      sets[set] = &setLin[set * numPoints];

      for (int point = 0; point < numPoints; point++) {
         float x, y;

         cin >> x >> y;

         if (x > hix) {
            hix = x;
         }
         if (x < lox) {
            lox = x;
         }

         if (y > hiy) {
            hiy = y;
         }
         if (y < loy) {
            loy = y;
         }

         sets[set][point].x = x;
         sets[set][point].y = y;
      }
   }
}

void Splines::transform(int n, int e, int s, int w) {
   normx = (1 - e) / (lox - hix);
   normy = (1 - s) / (loy - hiy);
   float offx = ((1 - e) * -lox) / (lox - hix);
   float offy = ((1 - s) * -loy) / (loy - hiy);

   for (int set = 0; set < numSets; set++) {
      for (int point = 0; point < numPoints; point++) {
         sets[set][point].x = sets[set][point].x * normx + offx;
         sets[set][point].y = sets[set][point].y * normy + offy;
      }
   }
}

void Splines::generate() {
   //clock_t start, diff;
   struct timeval star, end;

   //start = clock();
   gettimeofday(&star, NULL);
   finVals = generatePoints(sets, numSets, numPoints, resolution);
   gettimeofday(&end, NULL);
   //diff = clock() - start;
   
   if (timing) {
      //double sec, ms;
      double delta;
      
      //Uses CLOCKS_PER_SEC to determine the system's clock rate
      //sec = diff/((double)CLOCKS_PER_SEC);
      //ms = diff/(CLOCKS_PER_SEC / 1000);
      delta = ((end.tv_sec  - star.tv_sec) * 1000000u + end.tv_usec - star.tv_usec) / 1.e6;
      //cout << endl << "Time taken: " << sec << " seconds" << endl;
      //cout << "Time taken: " << ms << " miliseconds" << endl;
      //cout << endl << "Seconds" << (end.tv_sec  - star.tv_sec) << " usec " << (end.tv_usec - star.tv_usec);
      cout << endl << "New Timing Results: " << delta << "seconds." << endl;
   }
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
