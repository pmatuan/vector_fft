TARGET=main
OBJECTS=

CPPFLAGS=-std=c++11 -O3 -Wall -mavx2
LDFLAGS=-lm -pthread

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
