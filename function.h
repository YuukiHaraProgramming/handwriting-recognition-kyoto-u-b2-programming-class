/* Include guard */
#ifndef _FUNCTION_H_
#define _FUNCTION_H_

#include "coefficientDatabase.h"

/*関数のプロトタイプ宣言を行う*/
int inference6(const fc_matrix *, const float *, output_vector *);
void NN(fc_matrix *, float *, unsigned char *, int , float *, unsigned char *, int, int, int);
void save(const char *, int, int, const float *, const float *);
void load(const char *, int, int, float *, float *);

#endif