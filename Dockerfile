# Olá mundo com as dependências do transcritor em C++.
#   docker build -t ola-transcritor-cpp .
#   docker run --rm [--gpus all] ola-transcritor-cpp

# CUDA 12.8, TensorRT 10.16 e o kernel
FROM python@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS cuda
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential \
 && curl -fsSLo /tmp/keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
 && echo "e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d  /tmp/keyring.deb" | sha256sum -c - \
 && dpkg -i /tmp/keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-nvcc-12-8=12.8.93-1 cuda-cudart-dev-12-8=12.8.90-1 \
      libcublas-dev-12-8=12.8.4.1-1 libcufft-dev-12-8=11.3.3.83-1 \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/trt.whl https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-10.16.1.11-py3-none-manylinux_2_28_x86_64.whl \
 && echo "8e45036efeb964d323231544442a73619201136ccc84392560254cc8f0d516e4  /tmp/trt.whl" | sha256sum -c - \
 && python -m zipfile -e /tmp/trt.whl /tmp/trt \
 && C=/usr/local/cuda-12.8/targets/x86_64-linux \
 && mkdir -p /opt/lib /opt/include \
 && cd /tmp/trt/tensorrt_libs \
 && cp -a libnvinfer.so.10 libnvonnxparser.so.10 libnvinfer_builder_resource_ptx* libnvinfer_builder_resource_sm* /opt/lib/ \
 && cp -a $C/lib/libcudart.so* $C/lib/libcublas*.so* $C/lib/libcufft.so* /opt/lib/ \
 && cd / \
 && ln -s libnvinfer.so.10 /opt/lib/libnvinfer.so \
 && ln -s libnvonnxparser.so.10 /opt/lib/libnvonnxparser.so \
 && cp -a $C/include/. /opt/include/ \
 && curl -fsSL https://github.com/NVIDIA/TensorRT/archive/52399f555c2f80cb690a4a558b604e1a5f227e7c.tar.gz \
    | tar -xz -C /opt/include --strip-components=2 TensorRT-52399f555c2f80cb690a4a558b604e1a5f227e7c/include \
 && curl -fsSLo /opt/include/NvOnnxParser.h https://raw.githubusercontent.com/onnx/onnx-tensorrt/6bde6596eca84abf9afe9b4d21c18c32800fd3ad/NvOnnxParser.h \
 && echo "18735e1b92addcf2a9fd1480721e41d4ba86b39c4e1bb05a3bdde2c8638f6d49  /opt/include/NvOnnxParser.h" | sha256sum -c - \
 && rm -rf /tmp/trt /tmp/trt.whl

COPY ola.cu /src/
RUN /usr/local/cuda-12.8/bin/nvcc -O3 -arch=sm_75 -Xcompiler -fPIC -c /src/ola.cu -o /opt/ola_cuda.o

# torch de CPU (o transcritor usa no build para montar o pacote do modelo)
FROM python@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS pacote
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu \
 && pip install --no-cache-dir safetensors==0.8.0 numpy==2.5.4 \
 && python -c "import torch, safetensors, numpy; print('torch', torch.__version__, safetensors.__version__, numpy.__version__)" > /versoes.txt

# gRPC 1.85.1 compilado em Release e o programa
FROM cgr.dev/chainguard/wolfi-base@sha256:24f549a7fcace88e69b9557941168c74645688ecc4acf7fb6d734a6b74e9019d AS compilar
RUN apk add --no-cache build-base=1-r10 cmake=4.4.5-r0 samurai=1.3-r3 pkgconf=3.0.9-r0 curl=8.23.0-r3 patch=2.8.1-r1 \
      abseil-cpp-dev=20260817.0-r7 protobuf-dev=36.2-r1 protobuf-static=36.2-r1 protoc=36.2-r1 c-ares-dev=1.34.9-r0 \
      re2-dev=2025.11.05-r10 openssl-4.0-dev=4.0.3-r8 zlib-dev=1.3.2.1_rc20260921-r2 nlohmann-json-dev=3.12.0-r5

COPY ssl-ctx-null-check.patch /src/
RUN cd /src \
 && curl -fsSLo grpc.tar.gz https://github.com/grpc/grpc/archive/refs/tags/v1.85.1.tar.gz \
 && echo "90d8098ae0b7a1cdbccd2162193bb9d162d64c4e332a89daa4317a49fe92f386  grpc.tar.gz" | sha256sum -c - \
 && tar -xzf grpc.tar.gz \
 && patch -d grpc-1.85.1 -p1 < ssl-ctx-null-check.patch \
 && rm -rf grpc-1.85.1/third_party/grpc-proto \
 && GCC_SPEC_FILE=no-hardening.spec cmake -S grpc-1.85.1 -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/grpc \
      -DCMAKE_C_FLAGS=-DOPENSSL_NO_ENGINE -DCMAKE_CXX_FLAGS=-DOPENSSL_NO_ENGINE \
      -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_ABSL_PROVIDER=package -DgRPC_CARES_PROVIDER=package \
      -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_RE2_PROVIDER=package -DgRPC_SSL_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package \
 && GCC_SPEC_FILE=no-hardening.spec cmake --build build \
 && cmake --install build \
 && rm -rf /src

ENV PATH=/opt/grpc/bin:$PATH PKG_CONFIG_PATH=/opt/grpc/lib/pkgconfig LD_LIBRARY_PATH=/opt/grpc/lib
COPY --from=cuda /opt /opt
COPY ola.cpp ola.proto /src/
RUN cd /src \
 && protoc --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=/opt/grpc/bin/grpc_cpp_plugin ola.proto \
 && g++ -O2 -std=c++17 -I. -I/opt/include $(pkg-config --cflags grpc++ protobuf) \
      ola.cpp ola.pb.cc ola.grpc.pb.cc /opt/ola_cuda.o -L/opt/lib -Wl,-rpath-link,/opt/lib -lnvinfer -lnvonnxparser -lcudart -lcublas -lcufft \
      $(pkg-config --libs grpc++ protobuf) -lcrypto -lz -o /opt/ola

# a imagem
FROM cgr.dev/chainguard/wolfi-base@sha256:24f549a7fcace88e69b9557941168c74645688ecc4acf7fb6d734a6b74e9019d
RUN apk add --no-cache libabsl2608.0.0=20260817.0-r7 libprotobuf36.2.0=36.2-r1 libprotoc36.2.0=36.2-r1 c-ares=1.34.9-r0 \
      libre2-11=2025.11.05-r10 openssl-4.0-libssl=4.0.3-r8 openssl-4.0-libcrypto=4.0.3-r8 zlib=1.3.2.1_rc20260921-r2
ENV LD_LIBRARY_PATH=/opt/lib:/usr/local/nvidia/lib64:/usr/lib64 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY --from=cuda /opt/lib /opt/lib
COPY --from=compilar /opt/grpc/lib/*.so* /opt/lib/
COPY --from=compilar /opt/ola /app/ola
COPY --from=pacote /versoes.txt /app/
USER 65532
RUN cat /app/versoes.txt && /app/ola
CMD ["/app/ola"]
