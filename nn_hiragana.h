#ifndef NN_HIRAGANA_H
#define NN_HIRAGANA_H

#include "nn.h"

// NDL HIRAGANAのファイルを読み込む
// https://github.com/ndl-lab/hiragana_mojigazo
void load_hiragana(float ** train_x, unsigned char ** train_y, int * train_count,
                   float ** test_x, unsigned char ** test_y, int * test_count,
                   int * width, int * height) {
  assert(train_x != NULL);
  *train_x = load_mnist_image("hiragana-train-images-idx3-ubyte", width, height, train_count);
  assert(*train_x != NULL);
  assert(*width == 28);
  assert(*height == 28);
  assert(*train_count == 34500);

  assert(train_y != NULL);
  *train_y = load_mnist_label("hiragana-train-labels-idx1-ubyte", train_count);
  assert(*train_y != NULL);
  assert(*train_count == 34500);

  assert(test_x != NULL);
  *test_x = load_mnist_image("hiragana-test-images-idx3-ubyte", width, height, test_count);
  assert(*test_x != NULL);
  assert(*width == 28);
  assert(*height == 28);
  assert(*test_count == 11500);

  assert(test_y != NULL);
  *test_y = load_mnist_label("hiragana-test-labels-idx1-ubyte", test_count);
  assert(*test_y != NULL);
  assert(*test_count == 11500);
}

#endif // NN_HIRAGANA_H