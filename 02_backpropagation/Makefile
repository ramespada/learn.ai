CC := gcc
CSTD := -std=c11
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic $(CSTD)
LDFLAGS ?=
LDLIBS := -lm

TARGET := mlp_xor.x
SRCS := mlp_xor.c
OBJS := $(SRCS:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
