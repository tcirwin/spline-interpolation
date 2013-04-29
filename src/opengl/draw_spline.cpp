#include <GL/glut.h>
#include <iostream>
#include "../process_curve.h"

using namespace std;

void processCurve(int set, int numPoints, int res, Point* vals) {
   for(int dimPos = 0; dimPos < numPoints - 2; dimPos++){
      for (int i = 0; i < res; i++) {
         glBegin(GL_LINES);
            glVertex3f(vals[dimPos * res + i].x, vals[dimPos * res + i].y, 1);
            glVertex3f(vals[dimPos * res + i + 1].x, vals[dimPos * res + i + 1].y, 1);
         glEnd();
      }
   }
}
