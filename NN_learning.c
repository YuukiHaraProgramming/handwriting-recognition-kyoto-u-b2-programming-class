#include <time.h>
#include "nn.h"

#include "function.h"
#include "coefficientDatabase.h"

/* メモリ解放時にダブルフリーを防ぐ */
/*---------------------------------------
manual of free()
If ptr is NULL, no operation is performed.
---------------------------------------*/
#define SAFE_FREE(ptr) if(ptr != NULL ) { free(ptr); ptr = NULL; }


int main() {

  float *train_x = NULL;
  unsigned char *train_y = NULL;
  int train_count = -1;
  float *test_x = NULL;
  unsigned char *test_y = NULL;
  int test_count = -1;
  int width = -1;
  int height = -1;

  load_mnist(&train_x, &train_y, &train_count,
            &test_x, &test_y, &test_count,
            &width, &height);


  /* 乱数を初期化する */
  srand(time(NULL));

  /* 係数行列を定義 */
  fc_matrix *coef_matrix = malloc(sizeof(fc_matrix));

  /* 上で定義した係数行列に対して、NNを組む。 */
  NN(coef_matrix, train_x, train_y, train_count, test_x, test_y, test_count, width, height);

  save("fc1.dat", 50, 784, coef_matrix->A1, coef_matrix->b1);
  save("fc2.dat", 100, 50, coef_matrix->A2, coef_matrix->b2);
  save("fc3.dat", 10, 100, coef_matrix->A3, coef_matrix->b3);

  SAFE_FREE(coef_matrix);

  return 0;
}