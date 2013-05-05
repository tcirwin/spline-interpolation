#include <stdlib.h>
#include "init_curves.h"
#include "process_curve.h"
#include "spline.h"
#include "timing.h"

using namespace Splines;

int timing;

int main(int argc, char** argv) {  
   if (argc == 2)
      timing = atoi(argv[1]);
   else
      timing = 0;

   init();
   transform(0, 400, 300, 0);
   generate();
   iterate(processCurve);
   cleanup();
}
