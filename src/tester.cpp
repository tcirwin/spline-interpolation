#include "init_curves.h"
#include "process_curve.h"
#include "spline.h"

using namespace Splines;

int main(int argc, char** argv) {
   init();
   iterate(processCurve);
   cleanup();
}
