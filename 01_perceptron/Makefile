CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c11
TARGET = perceptron.x
SRC = slp.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
