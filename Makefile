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

validate-rain:
	python3 python/rain_plate_cpu.py \
	  --output outputs/rain_plate_v05 \
	  --validation-config config/rain_plate_validation.json

resolution-check:
	python3 python/resolution_convergence.py --out outputs/resolution_convergence

compare-reference:
	@test -n "$(REFERENCE)" || (echo "Usage: make compare-reference REFERENCE=/path/to/cpu_reference" && exit 2)
	python3 python/compare_reference.py \
	  --reference $(REFERENCE) \
	  --candidate outputs/rain_plate_v05 \
	  --out outputs/validation_compare

.PHONY: validate-rain resolution-check compare-reference

validate-output:
	python3 python/validate_outputs.py --output outputs/rain_plate_v06 --config config/rain_plate_validation.json

reproducibility-check:
	python3 python/check_reproducibility.py --out outputs/reproducibility

acceptance: validate-rain-v06 validate-output reproducibility-check

validate-rain-v06:
	python3 python/rain_plate_cpu.py \
	  --output outputs/rain_plate_v06 \
	  --validation-config config/rain_plate_validation.json

.PHONY: validate-output reproducibility-check acceptance validate-rain-v06

lsmps-cpu-smoke:
	PYTHONPATH=python python3 python/rain_plate_lsmps_cpu.py \
	  --output outputs/rain_plate_lsmps_cpu_smoke \
	  --steps 20 --inject-steps 16 --l0 0.016 --dt 0.001 \
	  --target-inflow-m3s 0.04 --no-shifting
	python3 python/validate_lsmps_cpu.py \
	  --output outputs/rain_plate_lsmps_cpu_smoke --min-valid-ratio 0.25

lsmps-cpu-reference:
	PYTHONPATH=python python3 python/rain_plate_lsmps_cpu.py \
	  --output outputs/rain_plate_lsmps_cpu \
	  --steps 80 --inject-steps 60 --l0 0.014 --dt 0.001 \
	  --target-inflow-m3s 0.04
	python3 python/validate_lsmps_cpu.py \
	  --output outputs/rain_plate_lsmps_cpu --min-valid-ratio 0.35

.PHONY: lsmps-cpu-smoke lsmps-cpu-reference
