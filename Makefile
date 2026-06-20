NVCC = nvcc
CFLAGS = -O3 -arch=sm_80  # sm_80 = Ampere = A100
LIBS = -lcublas

all: benchmark

benchmark: src/benchmark.cu
	$(NVCC) $(CFLAGS) src/benchmark.cu -o benchmark $(LIBS)

clean:
	rm -f benchmark results/benchmark.csv results/benchmark_plots.png

run: benchmark
	mkdir -p results
	./benchmark

plot: run
	cd results && python3 plot.py
