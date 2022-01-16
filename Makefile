
BUILDDIR=build

CXX=clang++
CXXFLAGS=-std=c++17 -Wall -fPIC -O2 -g -fno-omit-frame-pointer -mavx -mavx2 -msse2 -msse4.1
INC:=-I. -isystem extern/pybind11/include -isystem extern/vectorclass $(shell python3-config --includes)
LD=clang++
LDFLAGS:= $(shell python3-config --ldflags) -shared -std=c++17

HDRS = compute_entropy.hh

PYTHON_LIBRARY_NAME=compute_entropy_python$(shell python3-config --extension-suffix)

all: Makefile make_dir $(PYTHON_LIBRARY_NAME)

$(BUILDDIR)/%.o: %.cc $(HDRS)
	$(CXX) -c $(INC) $(CXXFLAGS) -o $@ $<

$(PYTHON_LIBRARY_NAME): $(BUILDDIR)/compute_entropy_python.o $(BUILDDIR)/compute_entropy.o
	$(LD) $(LDFLAGS) -o $(BUILDDIR)/$(PYTHON_LIBRARY_NAME) $^
	cp $(BUILDDIR)/$(PYTHON_LIBRARY_NAME) .

test: test_wordle test_wordle_solver test_compute_entropy_python

test_wordle:
	python3 wordle_test.py

test_wordle_solver:
	python3 wordle_solver_test.py

test_compute_entropy_python: all
	python3 compute_entropy_python_test.py

make_dir:
	mkdir -p build

clean:
	rm -rf build/
	rm -f $(PYTHON_LIBRARY_NAME)
	rm -rf __pycache__
