/*
 * proj1 CAD 2021/2022 FCT/UNL
 * vad
 */
#include <assert.h>
#include <ctype.h>
#include <driver_types.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* read_ppm - read a PPM image ascii file
 *   returns pointer to data, dimensions and max colors (from PPM header)
 *   data format: sequence of width x height of 3 ints for R,G and B
 *   aborts on errors
 */
void read_ppm(FILE *f, int **img, int *width, int *height, int *maxcolors) {
  int count = 0;
  char ppm[10];
  int c;
  // header
  while ((c = fgetc(f)) != EOF && count < 4) {
    if (isspace(c))
      continue;
    if (c == '#') {
      while (fgetc(f) != '\n')
        ;
      continue;
    }
    ungetc(c, f);
    switch (count) {
    case 0:
      count += fscanf(f, "%2s", ppm);
      break;
    case 1:
      count += fscanf(f, "%d%d%d", width, height, maxcolors);
      break;
    case 2:
      count += fscanf(f, "%d%d", height, maxcolors);
      break;
    case 3:
      count += fscanf(f, "%d", maxcolors);
    }
  }
  assert(c != EOF);
  assert(strcmp("P3", ppm) == 0);
  // data
  int *data = *img = (int *)malloc(3 * (*width) * (*height) * sizeof(int));
  assert(img != NULL);
  int r, g, b, pos = 0;
  while (fscanf(f, "%d%d%d", &r, &g, &b) == 3) {
    data[pos++] = r;
    data[pos++] = g;
    data[pos++] = b;
  }
  assert(pos == 3 * (*width) * (*height));
}

/* write_ppm - write a PPM image ascii file
 */
void write_ppm(FILE *f, int *img, int width, int height, int maxcolors) {
  // header
  fprintf(f, "P3\n%d %d %d\n", width, height, maxcolors);
  // data
  for (int l = 0; l < height; l++) {
    for (int c = 0; c < width; c++) {
      int p = 3 * (l * width + c);
      fprintf(f, "%d %d %d  ", img[p], img[p + 1], img[p + 2]);
    }
    putc('\n', f);
  }
}

/* printImg - print to screen the content of img
 */
void printImg(int imgw, int imgh, const int *img) {
  for (int j = 0; j < imgh; j++) {
    for (int i = 0; i < imgw; i++) {
      int x = 3 * (i + j * imgw);
      printf("%d,%d,%d  ", img[x], img[x + 1], img[x + 2]);
    }
    putchar('\n');
  }
}

/* areaFilter - transform a point (line,col) with contributions from its
 * neighbours no change if filter={{0,0,0}, {0,1,0}, {0,0,0}};
 */
__global__ void areaFilter(int *out, int *img, int imgw, int imgh,
                           int filter[3][3]) {
  int line = blockIdx.x;
  int col = threadIdx.x;
  int filter2[3][3] = {{1, 2, 1}, // gaussian filter
                       {2, 4, 2},
                       {1, 2, 1}};
  int r = 0, g = 0, b = 0, n = 0;
  for (int l = line - 1; l < line + 2 && l < imgh; l++)
    for (int c = col - 1; c < col + 2 && c < imgw; c++)
      if (l >= 0 && c >= 0) {
        int idx = 3 * (l * imgw + c);
        int scale = filter2[l - line + 1][c - col + 1];
        r += scale * img[idx];
        g += scale * img[idx + 1];
        b += scale * img[idx + 2];
        n += scale;
      }
  int idx = 3 * (line * imgw + col);
  out[idx] = r / n; // normalize value
  out[idx + 1] = g / n;
  out[idx + 2] = b / n;
}
// [][][][][][]
// [][][][][][]
// [][][][][][]

void areaFilter(int *out, int *img, int line, int col, int width, int height,
                int filter[3][3]) {
  int r = 0, g = 0, b = 0, n = 0;
  for (int l = line - 1; l < line + 2 && l < height; l++)
    for (int c = col - 1; c < col + 2 && c < width; c++)
      if (l >= 0 && c >= 0) {
        int idx = 3 * (l * width + c);
        int scale = filter[l - line + 1][c - col + 1];
        r += scale * img[idx];
        g += scale * img[idx + 1];
        b += scale * img[idx + 2];
        n += scale;
      }
  int idx = 3 * (line * width + col);
  out[idx] = r / n; // normalize value
  out[idx + 1] = g / n;
  out[idx + 2] = b / n;
}

/* pointFilter - transform a point (line,col) with greyscale
 *          newcolor = alpha*grey(color) +(1-alpha)*color
 *          newcolor = color (no change) if alpha=0
 */
__global__ void pointFilter(int *out, int *img, int imgw, int imgh,
                            float alpha) {
  int line = blockIdx.x;
  int col = threadIdx.x;
  int r = 0, g = 0, b = 0;
  float grey;
  int idx = 3 * (line * imgw + col);
  r = img[idx];
  g = img[idx + 1];
  b = img[idx + 2];
  grey = alpha * (0.3 * r + 0.59 * g + 0.11 * b);
  out[idx] = (1 - alpha) * r + grey;
  out[idx + 1] = (1 - alpha) * g + grey;
  out[idx + 2] = (1 - alpha) * b + grey;
}

int main(int argc, char *argv[]) {
  int imgh, imgw, imgc;
  int *img;
  float alpha = 0.5;             // default value
  int filter[3][3] = {{1, 2, 1}, // gaussian filter
                      {2, 4, 2},
                      {1, 2, 1}};
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "usage: %s img.ppm [alpha]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (argc == 3) {
    alpha = atof(argv[2]);
  }
  FILE *f = fopen(argv[1], "r");
  if (f == NULL) {
    fprintf(stderr, "can't read file %s\n", argv[1]);
    return EXIT_FAILURE;
  }

  read_ppm(f, &img, &imgw, &imgh, &imgc);
  printf("PPM image %dx%dx%d\n", imgw, imgh, imgc);
  printImg(imgw, imgh, img);

  int *out = (int *)malloc(3 * imgw * imgh * sizeof(int));
  assert(out != NULL);

  clock_t t = clock();

  int *d_a;
  cudaMalloc(&d_a, 3 * imgw * imgh * sizeof(int));
  int *d_b;
  cudaMalloc(&d_b, 3 * imgw * imgh * sizeof(int));
  int *d_c;
  cudaMalloc(&d_c, 3 * imgw * imgh * sizeof(int));
  int *d_filter;

  if (d_a == NULL || d_b == NULL) {
    fprintf(stderr, "No GPU mem!\n");
    return EXIT_FAILURE;
  }

  cudaMemcpy(d_a, img, 3 * imgw * imgh * sizeof(int), cudaMemcpyHostToDevice);

  areaFilter<<<imgh, imgw>>>(d_b, d_a, imgw, imgh, filter);

  /*for (int l = 0; l < imgh; l++) {
    for (int c = 0; c < imgw; c++) {
      areaFilter(out, img, l, c, imgw, imgh, filter);
    }
  }*/
  cudaMemcpy(out, d_b, 3 * imgw * imgh * sizeof(int), cudaMemcpyDeviceToHost);
  printImg(imgw, imgh, out);

  // cudaMemcpy(d_a, out, 3 * imgw * imgh * sizeof(int),
  // cudaMemcpyHostToDevice);

  pointFilter<<<imgh, imgw>>>(d_c, d_b, imgw, imgh, alpha);

  cudaMemcpy(out, d_c, 3 * imgw * imgh * sizeof(int), cudaMemcpyDeviceToHost);
  t = clock() - t;
  printf("time %f ms\n", t / (double)(CLOCKS_PER_SEC / 1000));

  printImg(imgw, imgh, out);
  FILE *g = fopen("out.ppm", "w");
  write_ppm(g, out, imgw, imgh, imgc);
  fclose(g);
  return EXIT_SUCCESS;
}
