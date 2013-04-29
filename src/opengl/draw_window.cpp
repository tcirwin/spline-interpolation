#include <GL/glut.h>
#include "../init_curves.h"
#include "../process_curve.h"

void init();
void display();

int main(int argc, char** argv) {
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
   glutInitWindowSize(350, 450);
   glutInitWindowPosition(200, 200);

   glutCreateWindow("Spline Interpolation Demo");
   init();

   glutDisplayFunc(display);
   glutMainLoop();

   return 0;
}

void init() {
   /*  select clearing (background) color       */
   glClearColor (0.0, 0.0, 0.0, 0.0);

   /*  initialize viewing values  */
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

   Splines::init();
}

void display() {
   /*  clear all pixels  */
    glClear (GL_COLOR_BUFFER_BIT);

    glColor3f (1.0, 1.0, 1.0);
    Splines::iterate(processCurve);
 
/*  don't wait! 
 *  start processing buffered OpenGL routines
 */
    glFlush ();
}
