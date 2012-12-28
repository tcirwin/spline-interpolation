#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>

float random_float(float low, float high);

using namespace std;

int main(int argc, char* argv[]) {
   assert(argc == 4); 

   int resolution = atoi(argv[1]);
   int numPoints = atoi(argv[2]);
   int numSets = atoi(argv[3]);

   cout << resolution << " " << numPoints << " " << numSets << endl;

   for (int set = 0; set < numSets; set++) {
      for (int point = 0; point < numPoints; point++) {
         cout << (point + 1) << " " << random_float(-100, 100) << endl;
      }
   }
}

float random_float(float low, float high) {
   int val = rand();
   float variance = ((float) rand()) / ((float) RAND_MAX);

   return ((float) (val % ((int) (high - low)) + low)) + variance;
}
