#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int cmp(const void* left, const void* right) {
    float v = * (float*) left - * (float*) right;
    if (v < 1e-9) {
        return 0;
    } else {
        return v / fabs(v);
    }
}

int main(int argc, const char* argv[]) {
    int num_elements = argc > 1 ? atoi(argv[1]) : 1000*1000;
    printf("%d\n", num_elements);
    float* a = malloc(num_elements * sizeof(*a));
    qsort((void*) a, num_elements, sizeof(*a), cmp);
    free(a);
    return 0;
}
