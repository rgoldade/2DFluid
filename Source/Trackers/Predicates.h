#ifndef PREDICATES_H
#define PREDICATES_H

double
exactinit(void); // call this before anything else

double
orient2d(const double *pa,
         const double *pb,
         const double *pc);

double
orient3d(const double *pa,
         const double *pb,
         const double *pc,
         const double *pd);

double
incircle(const double *pa,
         const double *pb,
         const double *pc,
         const double *pd);

double
insphere(const double *pa,
         const double *pb,
         const double *pc,
         const double *pd,
         const double *pe);

#endif
