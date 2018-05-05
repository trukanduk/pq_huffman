DEPDIR := .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

YAEL_FLAGS = -I$(YAEL_HOME) -lyael -L$(YAEL_HOME)/yael

CC = gcc
LD = $(CC)
CCFLAGS = -std=c99 -g $(YAEL_FLAGS)
PTHREAD_FLAGS=-pthread
LDFLAGS=-lm $(YAEL_FLAGS) $(PTHREAD_FLAGS)

COMPILE.c = $(CC) $(DEPFLAGS) $(CCFLAGS) -c
LINK.bin = $(LD) -o $@ $(DEPS_$@) $(LDFLAGS)
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

SOURCES := $(shell find . -name '*.c')
OBJECTS := $(SOURCES:%.c=%.o)
BINARIES := compute_nn_fast pq_encoder huffman_encoder huffman_decoder nn_huffman_estimator prepend_vecsl_meta

.PHONY: all clean

all: $(BINARIES)

clean:
	rm -f $(BINARIES) $(OBJECTS)

DEPS_compute_nn_fast = compute_nn_fast.o concurrent_queue.o misc.o fast_nn_temp_file.o vecs_io.o
compute_nn_fast: $(DEPS_compute_nn_fast)
	$(LINK.bin)

DEPS_pq_encoder = pq_encoder.o misc.o vecs_io.o
pq_encoder: $(DEPS_pq_encoder)
	$(LINK.bin)

DEPS_huffman_encoder = huffman_encoder.o huffman_encode.o huffman_decode.o huffman_codebook.o bitstream.o stats.o misc.o vecs_io.o
huffman_encoder: $(DEPS_huffman_encoder)
	$(LINK.bin)

DEPS_huffman_decoder = huffman_decoder.o huffman_encode.o huffman_decode.o huffman_codebook.o bitstream.o stats.o misc.o vecs_io.o
huffman_decoder: $(DEPS_huffman_decoder)
	$(LINK.bin)

DEPS_nn_huffman_estimator = nn_huffman_estimator.o huffman_encode.o huffman_decode.o huffman_codebook.o bitstream.o stats.o misc.o vecs_io.o
nn_huffman_estimator: $(DEPS_nn_huffman_estimator)
	$(LINK.bin)

DEPS_prepend_vecsl_meta = prepend_vecsl_meta.o vecs_io.o
prepend_vecsl_meta: $(DEPS_prepend_vecsl_meta)
	$(LINK.bin)

%.o : %.c
%.o : %.c $(DEPDIR)/%.d
	$(COMPILE.c) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SOURCES))))
