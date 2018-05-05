#ifndef _VECS_IO_H
#define _VECS_IO_H

#include <stdio.h>

#include "misc.h"

byte_t* load_vecs_light_filename(const char* filename, size_t element_size, long long num_elements);
byte_t* load_vecs_light_file(FILE* file, size_t element_size, long long num_elements);

#endif //_VECS_IO_H
