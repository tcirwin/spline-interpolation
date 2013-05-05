#ifdef __APPLE__
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
#endif

#include <stdlib.h>
#include "../init_curves.h"
#include "../process_curve.h"
#include "../timing.h"

void init();
void display();

int timing;

int width = 640, height = 480;

int main(int argc, char** argv) {
   if (argc == 2)
      timing = atoi(argv[1]);
   else
      timing = 0;

   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
   glutInitWindowSize(width, height);
   glutInitWindowPosition(200, 200);

   glutCreateWindow("Spline Interpolation Demo");
   init();

   glutDisplayFunc(display);
   glutMainLoop();

   return 0;
}

void init() {
   /*  select clearing (background) color       */
   glClearColor(1.0, 1.0, 1.0, 0.0);
   //glEnable(GL_LINE_SMOOTH);
   //glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

   /*  initialize viewing values  */
   //glMatrixMode(GL_PROJECTION);
   //glLoadIdentity();
   glOrtho(-50.0, width, -50.0, height, -2.0, 2.0);

   Splines::init();
   Splines::transform(0, width - 50, height - 50, 0);
   Splines::generate();
}

void display() {
   /*  clear all pixels  */
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(0.0, 0.0, 0.0);
    Splines::iterate(processCurve);
 
/*  don't wait! 
 *  start processing buffered OpenGL routines
 */
    glFlush ();
}
