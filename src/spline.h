#ifndef SPLINE_H
#define SPLINE_H

typedef struct Point {
   float x;
   float y;
} Point;

typedef struct CubicCurve {
   Point p1;
   Point p2;
   float z1;
   float z2;
} CubicCurve;

void printSpline(CubicCurve);
CubicCurve* generateSplines(Point **, int, int);
float solveSpline(float, CubicCurve);
Point **generatePoints(Point **, int, int, int);

#endif // SPLINE_H
