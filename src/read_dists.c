#include <stdio.h>

#define NUM_NN 200
#define NUM_VECTORS 10000

typedef struct _temp_file_item {
    long long index;
    float dist;
} temp_file_item_t;

int main(int argc, const char* argv[]) {
    FILE* f = fopen(argv[1], "rb");
    double sum = 0.0;
    double min = 100500.0;
    double max = -100500.0;
    float data[NUM_NN];
    temp_file_item_t tdata[NUM_NN];
    for (int i = 0; i < NUM_VECTORS; ++i) {
#if 0
        fread(&tdata[0], NUM_NN, sizeof(*tdata), f);
        if (tdata[0].dist > 100) {
            printf("%d: ", i);
            for (int j = 0; j < NUM_NN; ++j) {
                printf("(%f, %lld) ", tdata[j].dist, tdata[j].index);
            }
            printf("\n");
        }
#endif
#if 1
        fread(&data[0], NUM_NN, sizeof(float), f);
        for (int j = 0; j < NUM_NN; ++j) {
            if (data[j] > 10) {
                printf("%d %d\n", i, j);
                break;
            }
            sum += data[j];
            // printf("%f\n", data[i]);
        }
#endif
#if 0
        if (i != 34) {
            continue;
        }

        printf("%d: ", i);
        for (int j = 1; j <= NUM_NN; ++j) {
            printf("%f ", data[j]);
        }
        printf("\n");
#endif
#if 0
        for (int j = 1; j <= NUM_NN; ++j) {
            if (data[j] < min) {
                min = data[j];
            }
            if (data[j] > max) {
                max = data[j];
            }
        }
#endif
    }
    printf("Sum dists %lf %lf %lf\n", sum, min, max);
    return 0;
}
