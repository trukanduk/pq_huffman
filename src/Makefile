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
LINK.c = $(LD) -o $@ $? $(LDFLAGS)
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

SOURCES := $(shell find . -name '*.c')
OBJECTS := $(SOURCES:%.c=%.o)
BINARIES := compute_nn_fast pq_encoder huffman_encoder huffman_decoder

.PHONY: all clean

all: $(BINARIES)

clean:
	rm -f $(BINARIES) $(OBJECTS)

compute_nn_fast: compute_nn_fast.o concurrent_queue.o misc.o fast_nn_temp_file.o
	$(LINK.c)

pq_encoder: pq_encoder.o
	$(LINK.c)

huffman_encoder: huffman_encoder.o huffman_encode.o huffman_decode.o huffman_codebook.o bitstream.o stats.o
	$(LINK.c)

huffman_decoder: huffman_decoder.o huffman_encode.o huffman_decode.o huffman_codebook.o bitstream.o stats.o
	$(LINK.c)

%.o : %.c
%.o : %.c $(DEPDIR)/%.d
	$(COMPILE.c) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SOURCES))))
