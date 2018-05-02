YAEL_FLAGS=-I$(YAEL_HOME) -lyael -L$(YAEL_HOME)/yael
LD_FLAGS=-lm $(YAEL_FLAGS)
CC_FLAGS=-std=c99 -g

all: compute_nn_fast pq_encoder simple_huffman_encoder

compute_nn_fast: compute_nn_fast.c
	gcc compute_nn_fast.c -o compute_nn_fast $(CC_FLAGS) -pthread $(LD_FLAGS)

pq_encoder: pq_encoder.c
	gcc pq_encoder.c -o pq_encoder $(CC_FLAGS) $(LD_FLAGS)

bitstream.o: bitstream.c bitstream.h
	gcc bitstream.c -o bitstream.o -c $(CC_FLAGS)

huffman_encode.o: huffman_encode.c huffman.h bitstream.h
	gcc huffman_encode.c -o huffman_encode.o -c $(CC_FLAGS)

huffman_decode.o: huffman_decode.c huffman.h bitstream.h
	gcc huffman_decode.c -o huffman_decode.o -c $(CC_FLAGS)

huffman_codebook.o: huffman_codebook.c huffman.h bitstream.h
	gcc huffman_codebook.c -o huffman_codebook.o -c $(CC_FLAGS)

stats.o: stats.c stats.h
	gcc stats.c -o stats.o -c $(CC_FLAGS)

simple_huffman_encoder: simple_huffman_encoder.o huffman_encode.o huffman_decode.o huffman_codebook.o bitstream.o stats.o
	gcc -o simple_huffman_encoder simple_huffman_encoder.o huffman_encode.o huffman_decode.o huffman_codebook.o bitstream.o stats.o $(LD_FLAGS)

simple_huffman_encoder.o: simple_huffman_encoder.c huffman.h bitstream.h
	gcc simple_huffman_encoder.c -c -o simple_huffman_encoder.o -c $(CC_FLAGS)

all_test: bitstream_test huffman_encode_test huffman_decode_test huffman_codebook_test

bitstream_test: bitstream.c bitstream.h
	gcc bitstream.c -o bitstream_test $(CC_FLAGS) -D_BITSTREAM_TEST

huffman_encode_test.o: huffman_encode.c huffman.h bitstream.h
	gcc huffman_encode.c -c -o huffman_encode_test.o $(CC_FLAGS) -D_HUFFMAN_ENCODE_TEST

huffman_encode_test: huffman_encode_test.o huffman_codebook.o bitstream.o
	gcc -o huffman_encode_test huffman_encode_test.o huffman_codebook.o bitstream.o

huffman_decode_test.o: huffman_decode.c huffman.h bitstream.h
	gcc huffman_decode.c -c -o huffman_decode_test.o $(CC_FLAGS) -D_HUFFMAN_DECODE_TEST

huffman_decode_test: huffman_decode_test.o huffman_codebook.o huffman_encode.o bitstream.o
	gcc -o huffman_decode_test huffman_decode_test.o huffman_codebook.o huffman_encode.o bitstream.o

huffman_codebook_test.o: huffman_codebook.c huffman.h bitstream.h
	gcc huffman_codebook.c -c -o huffman_codebook_test.o $(CC_FLAGS) -D_HUFFMAN_CODEBOOK_TEST

huffman_codebook_test: huffman_codebook_test.o huffman_encode.o bitstream.o
	gcc -o huffman_codebook_test huffman_codebook_test.o huffman_encode.o bitstream.o
