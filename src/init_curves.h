#include "process_curve.h"
#include "spline.h"

namespace Splines {
   void init();
   void iterate(
    void (*process)(int setNum, int numPoints, int resolution, Point* vals));
   void generate();
   void transform(int n, int e, int s, int w);
   void cleanup();
}
