/* Include guard */
#ifndef _COEFFICIENT_DATABASE_H_
#define _COEFFICIENT_DATABASE_H_

/*=============================================================================================
構造体として、それぞれに対応するサイズの配列を用意しておく。
fc_matrixでは、係数行列（ベクトル）や、それらの勾配等を定義していく。
output_vectorでは、それぞれの層での出力ベクトルを格納していくための配列を定義していく。
=============================================================================================*/
typedef struct {
  float A1[784 * 50], b1[50], A2[50 * 100], b2[100], A3[100 * 10], b3[10];
} fc_matrix;

typedef struct {
  float img[784], fc1[50], relu1[50], fc2[100], relu2[100], fc3[10], softmax[10];
} output_vector;

/*=====================
ここまでが構造体の定義
=====================*/

#endif