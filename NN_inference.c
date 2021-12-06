#include <stdio.h>
#include "nn.h"

#include "function.h"
#include "coefficientDatabase.h"

/* メモリ解放時にダブルフリーを防ぐ */
/*---------------------------------------
manual of free()
If ptr is NULL, no operation is performed.
---------------------------------------*/
#define SAFE_FREE(ptr) if(ptr != NULL ) { free(ptr); ptr = NULL; }


/*------------------------------------
argv[1]：fc1の係数を保存したファイル名
argv[2]：fc2の係数を保存したファイル名
argv[3]：fc3の係数を保存したファイル名
argv[4]：BMP画像の読み込み
------------------------------------*/
int main (int argc, char *argv[]) {

  if (argc != 5) {
    printf("Loading Error.\n");
  } else {

    fc_matrix *coef_matrix = malloc(sizeof(fc_matrix));
    float *x = load_mnist_bmp(argv[4]);

    load(argv[1], 50, 784, coef_matrix->A1, coef_matrix->b1);
    load(argv[2], 100, 50, coef_matrix->A2, coef_matrix->b2);
    load(argv[3], 10, 100, coef_matrix->A3, coef_matrix->b3);

    int ans = inference6(coef_matrix, x, NULL);

    printf("This number is %d.\n", ans);

    SAFE_FREE(coef_matrix);

  }

  return 0;
}