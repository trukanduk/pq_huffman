all: compute_nn_fast

compute_nn_fast: compute_nn_fast.c
	gcc compute_nn_fast.c -o compute_nn_fast -std=c99 -pthread -I$(YAEL_HOME) -lyael -L$(YAEL_HOME)/yael
