# ============================================================
# AdaComp: Adaptive Multi-Tier Compression Middleware for MPI
# ============================================================

CXX        = mpicxx
CXXFLAGS   = -O2 -fPIC -std=c++14 -Wall

# Library paths (adjust for your cluster)
SZ3_PREFIX = $(CURDIR)/SZ3/install
ZSTD_DIR   = $(CURDIR)/zstd/lib

INCLUDES   = -I$(SZ3_PREFIX)/include -I$(SZ3_PREFIX)/include/SZ3 -I$(ZSTD_DIR)
LIB_DIRS   = -L$(SZ3_PREFIX)/lib -L$(ZSTD_DIR)

.PHONY: all clean sync

all: libadacomp.so calibrate_adacomp test_adacomp

libadacomp.so: libadacomp.cpp
	$(CXX) $(CXXFLAGS) -shared -o $@ $< $(INCLUDES) $(LIB_DIRS) -lSZ3c -lzstd -ldl

calibrate_adacomp: calibrate_adacomp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(INCLUDES) $(LIB_DIRS) -lSZ3c -lzstd

test_adacomp: test_adacomp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f libadacomp.so calibrate_adacomp test_adacomp

sync:
	@echo "Syncing binaries to cluster nodes..."
	@for node in node1 node2; do \
		scp libadacomp.so test_adacomp calibrate_adacomp itadmin@$$node:$(CURDIR)/; \
	done
	@echo "Sync complete."
