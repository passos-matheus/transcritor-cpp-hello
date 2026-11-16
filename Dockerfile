# syntax=docker/dockerfile:1
# Olá mundo com TODAS as dependências do transcritor em C++ (o Dockerfile.cpp dele): as mesmas bases,
# as mesmas fontes de download, as mesmas versões e os mesmos sha256, e o mesmo jeito de compilar. No
# lugar do servidor e do modelo, um programa pequeno (ola.cpp + ola.cu) que usa cada uma delas. Serve
# para ver se a CI consegue baixar e compilar tudo antes de tentar a imagem de verdade (README.md).
#
# Contexto = a raiz deste repositório:
#
#   docker build -t ola-transcritor-cpp .
#   docker run --rm ola-transcritor-cpp                 # sem GPU: as versões e "sem GPU aqui"
#   docker run --rm --gpus all ola-transcritor-cpp      # com GPU: também o kernel e o TensorRT
#
# O programa já roda uma vez no fim do build (sem GPU): se uma lib não carregar, o build falha ali.

# ---------------------------------------------------------------- 1. CUDA, TensorRT e o .cu
FROM python@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS cuda

# CUDA 12.8 do repositório da NVIDIA para Debian 12, o keyring conferido por sha256
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && curl -fsSL -o /tmp/keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
 && echo "e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d  /tmp/keyring.deb" | sha256sum -c - \
 && dpkg -i /tmp/keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cuda-nvcc-12-8=12.8.93-1 cuda-crt-12-8=12.8.93-1 cuda-nvvm-12-8=12.8.93-1 \
      cuda-cudart-dev-12-8=12.8.90-1 cuda-cudart-12-8=12.8.90-1 \
      libcublas-dev-12-8=12.8.4.1-1 libcublas-12-8=12.8.4.1-1 libcufft-dev-12-8=11.3.3.83-1 libcufft-12-8=11.3.3.83-1 \
 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda-12.8/bin:$PATH

# TensorRT 10.16.1.11: as libs da roda tensorrt-cu12-libs (pypi.nvidia.com, sha256), os headers do
# repositório da NVIDIA (tag v10.16, pelo commit) e o NvOnnxParser.h do onnx-tensorrt (sha256)
RUN curl -fsSL -o /tmp/trt.whl https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-10.16.1.11-py3-none-manylinux_2_28_x86_64.whl \
 && echo "8e45036efeb964d323231544442a73619201136ccc84392560254cc8f0d516e4  /tmp/trt.whl" | sha256sum -c - \
 && mkdir -p /opt/runtime/lib /opt/trt-include \
 && python -c "import re, zipfile; z = zipfile.ZipFile('/tmp/trt.whl'); m = [n for n in z.namelist() if re.match(r'tensorrt_libs/lib(nvinfer|nvinfer_builder_resource_(ptx|sm[0-9]+)|nvonnxparser|nvinfer_plugin)\.so', n)]; print(m); [open('/opt/runtime/lib/' + n.split('/')[-1], 'wb').write(z.read(n)) for n in m]" \
 && rm /tmp/trt.whl \
 && ln -s libnvinfer.so.10 /opt/runtime/lib/libnvinfer.so \
 && ln -s libnvonnxparser.so.10 /opt/runtime/lib/libnvonnxparser.so \
 && ls -la /opt/runtime/lib \
 && git init -q /tmp/trt \
 && git -C /tmp/trt fetch -q --depth 1 https://github.com/NVIDIA/TensorRT 52399f555c2f80cb690a4a558b604e1a5f227e7c \
 && git -C /tmp/trt checkout -q FETCH_HEAD \
 && cp /tmp/trt/include/*.h /opt/trt-include/ \
 && rm -rf /tmp/trt \
 && curl -fsSL -o /opt/trt-include/NvOnnxParser.h https://raw.githubusercontent.com/onnx/onnx-tensorrt/6bde6596eca84abf9afe9b4d21c18c32800fd3ad/NvOnnxParser.h \
 && echo "18735e1b92addcf2a9fd1480721e41d4ba86b39c4e1bb05a3bdde2c8638f6d49  /opt/trt-include/NvOnnxParser.h" | sha256sum -c -

# as libs do CUDA (as mesmas do transcritor) e o CUDA para compilar o resto na Wolfi
RUN C=/usr/local/cuda-12.8/targets/x86_64-linux \
 && cp -a $C/lib/libcudart.so.12* $C/lib/libcublas.so.12* $C/lib/libcublasLt.so.12* $C/lib/libcufft.so.11* /opt/runtime/lib/ \
 && mkdir -p /opt/cuda/lib \
 && cp -a $C/include /opt/cuda/ \
 && cp -a $C/lib/libcudart.so* $C/lib/libcublas.so* $C/lib/libcublasLt.so* $C/lib/libcufft.so* /opt/cuda/lib/

# o .cu: as mesmas opções dos .cu do transcritor (Release, SASS + PTX de cada arquitetura)
COPY ola.cu /src/ola.cu
WORKDIR /build/cu
RUN nvcc -O3 -DNDEBUG -std=c++17 -Xcompiler -fPIC \
      '--generate-code=arch=compute_75,code=[compute_75,sm_75]' '--generate-code=arch=compute_80,code=[compute_80,sm_80]' \
      '--generate-code=arch=compute_86,code=[compute_86,sm_86]' '--generate-code=arch=compute_89,code=[compute_89,sm_89]' \
      '--generate-code=arch=compute_90,code=[compute_90,sm_90]' \
      -c /src/ola.cu -o ola.o \
 && ar rcs /build/libola_cuda.a ola.o

# ---------------------------------------------------------------- 2. o que o estágio do pacote baixa
# (no transcritor, o torch de CPU monta o pacote do modelo; aqui só instala e mostra as versões)
FROM python@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS pacote

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu \
 && pip install --no-cache-dir safetensors==0.8.0 numpy==2.5.4 \
 && mkdir -p /out \
 && python -c "import torch, safetensors, numpy; print('torch', torch.__version__, '| safetensors', safetensors.__version__, '| numpy', numpy.__version__)" | tee /out/versoes.txt

# ---------------------------------------------------------------- 3. o C++ na Wolfi
FROM cgr.dev/chainguard/wolfi-base@sha256:24f549a7fcace88e69b9557941168c74645688ecc4acf7fb6d734a6b74e9019d AS compilar

RUN apk add --no-cache build-base=1-r10 cmake=4.4.5-r0 samurai=1.3-r3 pkgconf=3.0.9-r0 \
      curl=8.23.0-r3 patch=2.8.1-r1 abseil-cpp-dev=20260817.0-r7 protobuf-dev=36.2-r1 protobuf-static=36.2-r1 \
      protoc=36.2-r1 c-ares-dev=1.34.9-r0 re2-dev=2025.11.05-r10 openssl-4.0-dev=4.0.3-r8 \
      zlib-dev=1.3.2.1_rc20260921-r2 nlohmann-json-dev=3.12.0-r5

# o gRPC 1.85.1 da fonte em Release, igual ao transcritor (o do apk da Wolfi vem sem -O2 nem NDEBUG):
# a receita da Wolfi (o mesmo patch, as dependências do sistema, OPENSSL_NO_ENGINE, C++17, o gcc sem o
# spec de hardening, que quebra o constexpr dos headers do abseil). É a parte demorada (~20 min com 8 vCPU).
COPY ssl-ctx-null-check.patch /src/grpc.patch
WORKDIR /src
RUN curl -fsSL -o grpc.tar.gz https://github.com/grpc/grpc/archive/refs/tags/v1.85.1.tar.gz \
 && echo "90d8098ae0b7a1cdbccd2162193bb9d162d64c4e332a89daa4317a49fe92f386  grpc.tar.gz" | sha256sum -c - \
 && echo "d025cbd12111cdec676f06aa32814d7636a29a73c5f2df8282afaffb5dab7cb2  grpc.patch" | sha256sum -c - \
 && tar -xzf grpc.tar.gz \
 && patch -d grpc-1.85.1 -p1 < grpc.patch \
 && rm -rf grpc.tar.gz grpc-1.85.1/third_party/grpc-proto \
 && export GCC_SPEC_FILE=no-hardening.spec \
 && cmake -S grpc-1.85.1 -B /build/grpc -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/grpc \
      -DCMAKE_C_FLAGS=-DOPENSSL_NO_ENGINE -DCMAKE_CXX_FLAGS=-DOPENSSL_NO_ENGINE \
      -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_BUILD_CODEGEN=ON -DgRPC_BACKWARDS_COMPATIBILITY_MODE=OFF \
      -DgRPC_ABSL_PROVIDER=package -DgRPC_CARES_PROVIDER=package -DgRPC_PROTOBUF_PROVIDER=package \
      -DgRPC_RE2_PROVIDER=package -DgRPC_SSL_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package \
      -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
      -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF \
 && cmake --build /build/grpc -j"$(nproc)" \
 && cmake --install /build/grpc \
 && rm -rf /build/grpc grpc-1.85.1
ENV PATH=/opt/grpc/bin:$PATH \
    PKG_CONFIG_PATH=/opt/grpc/lib/pkgconfig \
    LD_LIBRARY_PATH=/opt/grpc/lib
COPY --from=cuda /opt/cuda /opt/cuda
COPY --from=cuda /opt/trt-include /opt/trt-include
COPY --from=cuda /opt/runtime/lib /opt/runtime/lib
COPY --from=cuda /build/libola_cuda.a /build/libola_cuda.a
COPY ola.proto ola.cpp /src/ola/
# o código do .proto (protoc + grpc_cpp_plugin), o programa ligado como o transcritor-srv, e as libs do
# gRPC que ele usa em /opt/grpc-rt (com o nome do soname)
WORKDIR /build/ola
RUN protoc -I/src/ola --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc="$(command -v grpc_cpp_plugin)" /src/ola/ola.proto \
 && g++ -O2 -std=c++17 -Wall -Wno-deprecated-declarations -I. -I/opt/cuda/include -I/opt/trt-include \
      $(pkg-config --cflags grpc++ protobuf) \
      /src/ola/ola.cpp ola.pb.cc ola.grpc.pb.cc /build/libola_cuda.a \
      /opt/runtime/lib/libnvinfer.so /opt/runtime/lib/libnvonnxparser.so \
      /opt/cuda/lib/libcudart.so /opt/cuda/lib/libcublas.so /opt/cuda/lib/libcufft.so \
      $(pkg-config --libs grpc++ protobuf) -lcrypto -lz -ldl -lpthread -o ola \
 && strip ola \
 && LD_LIBRARY_PATH=/opt/runtime/lib:/opt/grpc/lib LD_TRACE_LOADED_OBJECTS=1 ./ola > libs.txt \
 && cat libs.txt \
 && ! grep "not found" libs.txt \
 && grep -q "/opt/grpc/lib/libgrpc++" libs.txt \
 && mkdir -p /opt/grpc-rt \
 && awk '$3 ~ "^/opt/grpc/lib/" {print $3}' libs.txt | xargs -I{} cp -L {} /opt/grpc-rt/ \
 && ls -la /opt/grpc-rt

# ---------------------------------------------------------------- 4. a imagem
FROM cgr.dev/chainguard/wolfi-base@sha256:24f549a7fcace88e69b9557941168c74645688ecc4acf7fb6d734a6b74e9019d

# as libs de runtime do transcritor (as mesmas versões) e o usuário do serviço (sem root)
RUN apk add --no-cache libabsl2608.0.0=20260817.0-r7 libprotobuf36.2.0=36.2-r1 libprotoc36.2.0=36.2-r1 \
      c-ares=1.34.9-r0 libre2-11=2025.11.05-r10 openssl-4.0-libssl=4.0.3-r8 openssl-4.0-libcrypto=4.0.3-r8 \
      zlib=1.3.2.1_rc20260921-r2 \
 && echo "transcritor:x:10001:10001:transcritor:/home/transcritor:/sbin/nologin" >> /etc/passwd \
 && echo "transcritor:x:10001:" >> /etc/group \
 && mkdir -p /app /home/transcritor \
 && chown 10001:10001 /home/transcritor

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/opt/transcritor/lib:/usr/local/nvidia/lib64:/usr/lib64 \
    HOME=/home/transcritor

WORKDIR /app
COPY --from=cuda /opt/runtime/lib /opt/transcritor/lib
COPY --from=compilar /opt/grpc-rt /opt/transcritor/lib
COPY --from=compilar /build/ola/ola /app/ola
COPY --from=pacote /out/versoes.txt /app/versoes-pacote.txt

USER 10001
# a prova, ainda no build (sem GPU): todas as libs carregam e o gRPC responde
RUN cat /app/versoes-pacote.txt && /app/ola
ENTRYPOINT ["/app/ola"]
