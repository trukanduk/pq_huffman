all: compute_nn_fast pq

compute_nn_fast: compute_nn_fast.c
	gcc compute_nn_fast.c -o compute_nn_fast -std=c99 -pthread -I$(YAEL_HOME) -lyael -L$(YAEL_HOME)/yael

pq: pq.c
	gcc pq.c -o pq -std=c99 -lm -I$(YAEL_HOME) -lyael -L$(YAEL_HOME)/yael

bitstream.o: bitstream.c bitstream.h
	gcc bitstream.c -o bitstream.o -c -std=c99

huffman_encode.o: huffman_encode.c huffman.h bitstream.h
	gcc huffman_encode.c -o huffman_encode.o -c -std=c99

huffman_decode.o: huffman_decode.c huffman.h bitstream.h
	gcc huffman_decode.c -o huffman_decode.o -c -std=c99

huffman_codebook.o: huffman_codebook.c huffman.h bitstream.h
	gcc huffman_codebook.c -o huffman_codebook.o -c -std=c99

all_test: bitstream_test huffman_encode_test huffman_decode_test huffman_codebook_test

bitstream_test: bitstream.c bitstream.h
	gcc bitstream.c -o bitstream_test -std=c99 -D_BITSTREAM_TEST

huffman_encode_test.o: huffman_encode.c huffman.h bitstream.h
	gcc huffman_encode.c -c -o huffman_encode_test.o -std=c99 -D_HUFFMAN_ENCODE_TEST

huffman_encode_test: huffman_encode_test.o huffman_codebook.o bitstream.o
	gcc -o huffman_encode_test huffman_encode_test.o huffman_codebook.o bitstream.o

huffman_decode_test.o: huffman_decode.c huffman.h bitstream.h
	gcc huffman_decode.c -c -o huffman_decode_test.o -std=c99 -D_HUFFMAN_DECODE_TEST

huffman_decode_test: huffman_decode_test.o huffman_codebook.o huffman_encode.o bitstream.o
	gcc -o huffman_decode_test huffman_decode_test.o huffman_codebook.o huffman_encode.o bitstream.o

huffman_codebook_test.o: huffman_codebook.c huffman.h bitstream.h
	gcc huffman_codebook.c -c -o huffman_codebook_test.o -std=c99 -D_HUFFMAN_CODEBOOK_TEST

huffman_codebook_test: huffman_codebook_test.o huffman_encode.o bitstream.o
	gcc -o huffman_codebook_test huffman_codebook_test.o huffman_encode.o bitstream.o
