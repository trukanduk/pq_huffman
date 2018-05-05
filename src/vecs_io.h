#ifndef _VECS_IO_H
#define _VECS_IO_H

#include <stdio.h>

#include "misc.h"

byte_t* load_vecs_light_filename(const char* filename, size_t element_size,
                                 long long* num_elements_out, int* num_dimensions_out);
byte_t* load_vecs_light_file(FILE* file, size_t element_size,
                             long long* num_elements_out, int* num_dimensions_out);

void load_vecs_light_meta_filename(const char* filename, long long* num_elements_out,
                                   int* num_dimensions_out);
void load_vecs_light_meta_file(FILE* file, long long* num_elements_out,
                               int* num_dimensions_out);
long long load_vecs_num_vectors_filename(const char* filename);
int load_vecs_num_dimensions_filename(const char* filename);

void save_vecs_light_meta_file(FILE* file, long long num_elements, int num_dimensions);


#endif //_VECS_IO_H
