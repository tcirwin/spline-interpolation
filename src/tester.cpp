#include "init_curves.h"
#include "process_curve.h"
#include "spline.h"

using namespace Splines;

int main(int argc, char** argv) {
   init();
   transform(0, 400, 300, 0);
   generate();
   iterate(processCurve);
   cleanup();
}
