USAGE
=====
 - Compile: `make`

 - Plotting Splines: `./windowtest < [data file]`
 - Write Spline Points to Console: `./splinetest < [data file]`
 - Sample data files available: `/tests/*.in`.



Spline Interpolation
====================
Uses the Jacobi method to calculate natural cubic splines for each set of points given, then calculates the specified number of intermediate points using the resolution given. Computes only a specified number of iterations of the Jacobi method, since checking for convergence is much more difficult and relatively unnecessary (see solve_system.cu).

API
===
    Point** generatePoints(Point** sets, int numSets, int numPoints, int resolution);

Generates the spline curves from the set of points given, with the following options:
 - `numSets`: the number of data lines that are being passed in (length of the first dimension in the `sets` pointer). Must be > 0.
 - `numPoints`: the number of data points present in each spline; determines how many functions are needed to build each spline curve (length of the second dimension in the `sets` pointer). Must be > 0.
 - `resolution`: how many intermediate points will be drawn between each data point. Setting this higher will give a more defined curve, but significantly slows computation. 15 is a good number here. Must be > 0. Setting to 1 simply returns the `sets` pointer passed in.

`sets`: a double pointer to Point, a struct containing x and y.
