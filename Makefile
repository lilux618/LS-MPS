CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -Iinclude
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_80

all: bin/lsmps-bench

bin/lsmps-bench: src/main.cpp src/lsmps.cpp include/lsmps.hpp
	@mkdir -p bin results
	$(CXX) $(CXXFLAGS) src/main.cpp src/lsmps.cpp -o $@

cuda-objects: src/cuda_workload.cu
	@mkdir -p obj
	$(NVCC) $(NVCCFLAGS) -c src/cuda_workload.cu -o obj/cuda_workload.o

smoke: all
	./bin/lsmps-bench config/smoke.cfg results

suite: all
	bash scripts/run_suite.sh

clean:
	rm -rf bin obj results/*.json results/*.csv

.PHONY: all smoke suite cuda-objects clean

rain-demo:
	python3 python/rain_plate_cpu.py --output outputs/rain_plate

calibrate:
	python3 python/workload_calibration.py \
	  --target config/customer_car_rain_profile.example.json \
	  --baseline config/plate_rain_baseline_profile.json \
	  --out outputs/calibration

.PHONY: rain-demo calibrate
