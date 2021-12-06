#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "function.h"
#include "coefficientDatabase.h"

/* メモリ解放時にダブルフリーを防ぐ */
/*---------------------------------------
manual of free()
If ptr is NULL, no operation is performed.
---------------------------------------*/
#define SAFE_FREE(ptr) if(ptr != NULL ) { free(ptr); ptr = NULL; }


/*==================================
まず初めに基本的な関数を定義していく
==================================*/


/* 平均値mu、分散sigmaのガウス分布を用いて乱数を生成して返す関数 */
static float randGauss (float mu, float sigma) {
  /* 円周率piをarccos(-1)で定義 */
  const float pi = acosf(-1);

  /* 一様乱数の生成 */
  float uniform = (((float) rand() + 0.5) / (RAND_MAX + 1.0));

  /* Box–Muller's methodにより標準正規分布を求める */
  float z = sqrtf( -2.0 * logf(uniform) ) * sinf( 2.0 * pi * uniform );

/* 平均値mu、分散sigmaのガウス分布に変更して返す */
  return mu + sigma * z;
}

/* oの各要素をxで初期化する関数。nは要素数 */
static void init(int n, float x, float *o) {
  for (int i = 0; i < n; i++) {
    o[i] = x;
  }
}

/* <stdlib.h>ヘッダがRAND_MAXのために必要 */
/* [-1, 1]で初期化する関数 */
static void rand_init(int n, float *o) {
  for (int i = 0; i < n; i++) {
    /* 平均値0、分散√2/nの正規分布で生成する乱数で初期化 */
    o[i] = randGauss(0, sqrtf(2.0 / (float)n));
  }
}


/*--------------------------------------
初期化の仕方を選択する。
boolean == 0 のときはxを0で初期化。
boolean == 1 のときはxを[-1, 1]で初期化。
--------------------------------------*/
static void how_init (int n, float *x, int boolean) {
  if (boolean == 0) {
    init(n, 0, x);
  } else {
    rand_init(n, x);
  }
}


/* oの各要素にxの対応する各要素を加える関数 */
static void add(int n, const float *x, float *o) {
  for (int i = 0; i < n; i++) {
    o[i] += x[i];
  }
}

/* oの各要素をx倍する */
static void scale(int n, float x, float *o) {
  for (int i = 0; i < n; i++) {
    o[i] *= x;
  }
}

/* 配列の最大値をfloat型で返す関数 */
static float array_max(int n, const float *x) {
  float x_max = x[0];

  for (int i = 1; i < n; i++) {
    x_max = x[i] > x_max ? x[i] : x_max;
  }

  return x_max;
}

static void shuffle(int n, int *x) {
  for (int i = 0; i < n; i++) {
    int j = rand() % n;
    int temp = x[i];
    x[i] = x[j];
    x[j] = temp;
  }
}

/*==========================
ここまでが基本的な関数の定義
==========================*/


/*====================================================
2種類の構造体を引数に持つ、基本的な関数を定義していく。
同じような操作をまとめるための関数でもある。
====================================================*/

/* fc_matrixで生成する6つの配列をまとめて初期化する。初期化の種類はbooleanで指定することで選択可能 */
static void initMatrix6 (fc_matrix *matrix, int boolean) {
  how_init(784 * 50, matrix->A1, boolean);
  how_init(50, matrix->b1, boolean);
  how_init(50 * 100, matrix->A2, boolean);
  how_init(100, matrix->b2, boolean);
  how_init(100 * 10, matrix->A3, boolean);
  how_init(10, matrix->b3, boolean);
}

/* output_vectorで生成する6つの配列をまとめて0で初期化する。*/
static void initVector6_zero (output_vector *output) {
  how_init(50, output->fc1, 0);
  how_init(50, output->relu1, 0);
  how_init(100, output->fc2, 0);
  how_init(100, output->relu2, 0);
  how_init(10, output->fc3, 0);
  how_init(10, output->softmax, 0);
}

/* fc_matrixで生成する配列2種類を、まとめて各成分ごとに足し合わせる関数 */
static void add6(fc_matrix *matrix_add, fc_matrix *matrix_added) {
	add(50 * 784, matrix_add->A1, matrix_added->A1);
	add(50, matrix_add->b1, matrix_added->b1);
	add(100 * 50, matrix_add->A2, matrix_added->A2);
	add(100, matrix_add->b2, matrix_added->b2);
	add(10 * 100, matrix_add->A3, matrix_added->A3);
	add(10, matrix_add->b3, matrix_added->b3);
}

/* fc_matrixで生成する配列を、まとめて各成分scalar倍する関数 */
static void scale6(fc_matrix *matrix_added, float scalar) {
  scale(50 * 784, scalar, matrix_added->A1);
  scale(50, scalar, matrix_added->b1);
  scale(100 * 50, scalar, matrix_added->A2);
  scale(100, scalar, matrix_added->b2);
  scale(10 * 100, scalar, matrix_added->A3);
  scale(10, scalar, matrix_added->b3);
}

/*=============================================
ここまでが構造体を引数に持つ基本的な関数の定義
=============================================*/


/*==========================================
各層での計算をするための関数を定義していく。
==========================================*/

/*------------------------------------
fc層
Ax+bを計算し、値をoに格納する。Aはm×n。
------------------------------------*/
static void fc(int m, int n, const float *x, const float *A, const float *b, float *y) {

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      y[i] += A[n * i + j] * x[j];
    }
    y[i] += b[i];
  }
}

/*----
relu層
----*/
static void relu(int n, const float *x, float *y) {

  for (int i = 0; i < n; i++) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

/*--------
softmax層
--------*/
static void softmax(int n, const float *x, float *y) {
  float x_max = array_max(n, x);

  float exp_sum = 0;
  for (int i = 0; i < n; i++) {
    exp_sum += expf(x[i] - x_max);
  }

  for (int i = 0; i < n; i++) {
    y[i] = expf(x[i] - x_max) / exp_sum;
  }
}

/*---------------------
fc層でのbackward関数

x,dEdxの要素数はn
A,dEdAの要素数はm×n
dEdy,b,dEdbの要素数はm
---------------------*/
static void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A, float *dEdA, float *dEdb, float *dEdx) {

  for (int k = 0; k < m; k++) {
    for (int i = 0; i < n; i++) {
      dEdA[n * k + i] = dEdy[k] * x[i];
    }
  }

  for (int k = 0; k < m; k++) {
    dEdb[k] = dEdy[k];
  }

  for (int k = 0; k < n; k++) {
    for (int i = 0; i < m; i++) {
      dEdx[k] += A[n * i + k] * dEdy[i];
    }
  }
}

/*--------------------------------------------
relu層でのbackward関数

xの要素数とdEdyの要素数,dEdxの要素数はすべてn
--------------------------------------------*/
static void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx) {
  for (int i = 0; i < n; i++) {
    dEdx[i] = x[i] > 0 ? dEdy[i] : 0;
  }
}


/*----------------------------------
softmax層でのbackward関数

yの要素数とdEdxの要素数はどちらもn
----------------------------------*/
static void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx) {
  for (int k = 0; k < n; k++) {
    dEdx[k] = (k == (int)t) ? y[k] - 1 : y[k];
  }
}

/*=====================================
ここまでが各層での計算をするための関数
=====================================*/


/*--------------------------------------------------------------------------------------------
inference6では6層で推論を行い、且つ入力xがなんの数字であるかを推論した結果を整数値で返す関数。
coed_matrixはfc層での係数行列。xは入力値。各層ごとの出力をoutputに保存する。
--------------------------------------------------------------------------------------------*/
int inference6(const fc_matrix *coef_matrix, const float *x, output_vector *output) {

  /*-----------------------------------------------------------------------
  outputにNULLを渡したときは、outputをinference6内で確保する。
  ただし、NULLを渡すのは学習時ではなく、inference6を1回のみ実行する検証時。
  -----------------------------------------------------------------------*/
  if (output == NULL) {
    output = malloc(sizeof(output_vector));
  }

  /* まずは入力されたoutputを0で初期化する。 */
  initVector6_zero(output);

  /* 6層NN推論部分 */
  fc(50, 784, x, coef_matrix->A1, coef_matrix->b1, output->fc1);
  relu(50, output->fc1, output->relu1);
  fc(100, 50, output->relu1, coef_matrix->A2, coef_matrix->b2, output->fc2);
  relu(100, output->fc2, output->relu2);
  fc(10, 100, output->relu2, coef_matrix->A3, coef_matrix->b3, output->fc3);
  softmax(10, output->fc3, output->softmax);

  /* output->softmaxの中での最大値を返す。 */
  float y_max = output->softmax[0];
  int ans = 0;
  for (int i = 1; i < 10; i++) {
    if (output->softmax[i] > y_max) {
      y_max = output->softmax[i];
      ans = i;
    }
  }

  return ans;
}

/*-----------------------------------------------------------------------
backward6では6層で推論を行った後、逆伝播していきdE_matrixを計算していく。
coed_matrixはfc層での係数行列。xは入力値。tは正解の値。
dE_matrixがＥを各行列（ベクトル）の勾配行列（ベクトル）。
各層ごとの出力をoutputに保存する。dEdxは下の層へ勾配行列（ベクトル）。
-----------------------------------------------------------------------*/
static void backward6(const fc_matrix *coef_matrix, const float *x, unsigned char t, fc_matrix *dE_matrix, output_vector *output, output_vector *dEdx) {

  /* 推論を行う。各層での出力はoutputに保存。 */
  inference6(coef_matrix, x, output);

  /* 入力されたdEdxを0で初期化。 */
  initVector6_zero(dEdx);

  /* 6層NN逆伝播部分。 */
  softmaxwithloss_bwd(10, output->softmax, t, dEdx->fc3);
  fc_bwd(10, 100, output->relu2, dEdx->fc3, coef_matrix->A3, dE_matrix->A3, dE_matrix->b3, dEdx->relu2);
  relu_bwd(100, output->fc2, dEdx->relu2, dEdx->fc2);
	fc_bwd(100, 50, output->relu1, dEdx->fc2, coef_matrix->A2, dE_matrix->A2, dE_matrix->b2, dEdx->relu1);
	relu_bwd(50, output->fc1, dEdx->relu1, dEdx->fc1);
	fc_bwd(50, 784, x, dEdx->fc1, coef_matrix->A1, dE_matrix->A1, dE_matrix->b1, output->img);

}

/* 交差エントロピーを返すための関数。 */
static float cross_entropy_error(const float *y, int t) {
  return -logf(y[t] + 1e-7);
}

/*------------------------------------------------
以上で定義した関数を用いて、学習を行っていく関数。
------------------------------------------------*/
void NN (fc_matrix *coef_matrix, float *train_x, unsigned char *train_y, int train_count, float *test_x, unsigned char *test_y, int test_count, int width, int height) {

  /* 学習を行うために必要な定数の定義 */
  const int epoch = 10;
  const int n_minibatch = 100;
  const int N = train_count;
  const float eta = 0.1;

  /* 勾配行列（ベクトル）等を格納するための配列を動的に確保。 */
  fc_matrix *dE_matrix = malloc(sizeof(fc_matrix));
  fc_matrix *dE_matrix_ave = malloc(sizeof(fc_matrix));

  output_vector *output = malloc(sizeof(output_vector));
  output_vector *dEdx = malloc(sizeof(output_vector));

  /* 係数行列を[-1, 1]で初期化。 */
  initMatrix6(coef_matrix, 1);

  int *index = malloc(sizeof(int) * N);
  float E_loss;

  for (int cnt_epoch = 0; cnt_epoch < epoch; cnt_epoch++) {

    for (int i = 0; i < N; i++) {
      index[i] = i;
    }
    shuffle(N, index);

    for (int cnt = 0; cnt < N / n_minibatch; cnt++) {

      /* 平均勾配を0で初期化。 */
      initMatrix6(dE_matrix_ave, 0);

      for (int i = 0; i < n_minibatch; i++) {
        /* 勾配を0で初期化 */
        initMatrix6(dE_matrix, 0);

        /* 推論 + 逆伝播 */
        backward6(coef_matrix, train_x + 784 * index[i + cnt * n_minibatch], train_y[index[i + cnt * n_minibatch]], dE_matrix, output, dEdx);
        
        /* 計算された勾配を平均勾配に足していく。 */
        add6(dE_matrix, dE_matrix_ave);
      }

      /*--------------------------------------------------------------------------------------------
      n_minibatch回足された平均勾配をn_minibatchで割り（これが本来の平均勾配）、それに-etaをかける。
      そのあとに係数行列を更新する。
      --------------------------------------------------------------------------------------------*/
      scale6(dE_matrix_ave, -eta / (float)n_minibatch);
      add6(dE_matrix_ave, coef_matrix);
    }

    /* 計算された係数行列とテストデータを用いて、正答率と損失関数を計算していく。 */
    int sum = 0;
    output_vector *output_test = malloc(sizeof(output_vector));
    for (int i = 0; i < test_count; i++) {
      if (inference6(coef_matrix, test_x + 784 * i, output_test) == test_y[i]) {
        sum++;
      }
    }

    E_loss = 0;
    for (int i = 0; i < test_count; i++) {
      initVector6_zero(output_test);
      inference6(coef_matrix, test_x + 784 * i, output_test);
      E_loss += cross_entropy_error(output_test->softmax, test_y[i]);
    }
    E_loss /= test_count;

    SAFE_FREE(output_test);

    /* 得られた結果の出力 */
    printf("epoch %d:\n", cnt_epoch + 1);
    printf("%f%%\n", sum * 100.0 / test_count);
    printf("loss function test: %f\n", E_loss);
    printf("\n");
  }

  SAFE_FREE(index);
  SAFE_FREE(dE_matrix);
  SAFE_FREE(dE_matrix_ave);
}

/* 計算した係数行列（ベクトル）を保存する */
void save (const char *filename, int m, int n, const float *A, const float *b) {
  FILE *fp;
  fp = fopen(filename, "wb");
  fwrite(A, sizeof(A), m * n, fp);
  fwrite(b, sizeof(b), m, fp);
  fclose(fp);
}

/* 保存した係数行列（ベクトル）を読み込む */
void load (const char *filename, int m, int n, float *A, float *b) {
  FILE *fp;
  fp = fopen(filename, "rb");
  fread(A, sizeof(A), m * n, fp);
  fread(b, sizeof(b), m, fp);
  fclose(fp);
}
