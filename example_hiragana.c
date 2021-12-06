#include "nn_hiragana.h"

int main()
{
  float *train_x = NULL;
  unsigned char *train_y = NULL;
  int train_count = -1;

  float *test_x = NULL;
  unsigned char *test_y = NULL;
  int test_count = -1;

  int width = -1;
  int height = -1;

  load_hiragana(&train_x, &train_y, &train_count,
                &test_x, &test_y, &test_count,
                &width, &height);

  // これ以降，
  // 訓練データ train_x + 784*i (i=0,...,train_count-1), train_y[0]～train_y[train_count-1],
  // テストデータ test_x + 784*i (i=0,...,test_count-1), test_y[0]～test_y[test_count-1],
  // を使用することができる．

  // ラベルとひらがなの対応は hiragana-labels.txt を見ること

  return 0;
}
