# Olá mundo com as dependências do transcritor em C++

Um Dockerfile que baixa e compila **tudo o que o Dockerfile do transcritor em C++ usa**: as mesmas bases, as
mesmas fontes, as mesmas versões e os mesmos sha256. No lugar do servidor e do modelo há um programa
pequeno (`ola.cpp` + `ola.cu`). Serve para ver se a CI consegue baixar e compilar tudo antes de tentar
a imagem de verdade, que leva ~1,3 GB de modelo e o código do servidor.

Na raiz deste repositório:

    docker build -t ola-transcritor-cpp .
    docker run --rm ola-transcritor-cpp                 # sem GPU
    docker run --rm --gpus all ola-transcritor-cpp      # com GPU

O programa roda uma vez no fim do build. Se uma lib não carregar, ou se o gRPC não responder, o build
falha ali. A saída esperada, sem GPU:

    torch 2.11.0+cpu | safetensors 0.8.0 | numpy 2.5.4
    olá mundo: as dependências do transcritor em C++
    gRPC 1.85.1: "olá, mundo" pela porta NNNNN
    abseil LTS 20260817.0                               (se a versão do abseil for LTS)
    OpenSSL 4.0.3 ...: sha256("olá") = ...
    zlib 1.3.2.1...
    nlohmann-json 3.12.0: {"n":1,"ola":"mundo"}
    CUDA runtime 12.8, cuBLAS 12.8.4, cuFFT 11303
    TensorRT 10.16.1, parser de ONNX ...
    sem GPU aqui (...): o kernel e o builder do TensorRT ficam para uma máquina com GPU
    tudo ok

Com GPU, no lugar da linha "sem GPU", aparecem o kernel (soma 992) e o builder do TensorRT com o parser
de ONNX.

## O que a CI precisa alcançar

| de onde | o quê | estágio |
|---|---|---|
| Docker Hub (`registry-1.docker.io`) | a base `python@sha256:6a0e06f5…` | cuda, pacote |
| `cgr.dev` (Chainguard) | a base `wolfi-base@sha256:24f549a7…` | compilar, final |
| `deb.debian.org` | os pacotes do Debian (apt) | cuda |
| `developer.download.nvidia.com` | o keyring e o CUDA 12.8 (apt) | cuda |
| `pypi.nvidia.com` | a roda `tensorrt-cu12-libs` 10.16.1.11 | cuda |
| `github.com` | os headers do TensorRT (git) e o tarball do gRPC 1.85.1 (redireciona para `codeload.github.com`) | cuda, compilar |
| `raw.githubusercontent.com` | o `NvOnnxParser.h` | cuda |
| `download.pytorch.org` | o torch 2.11.0 de CPU | pacote |
| `pypi.org` e `files.pythonhosted.org` | o safetensors e o numpy | pacote |
| `packages.wolfi.dev` | os pacotes da Wolfi (apk) | compilar, final |

Tudo é x86_64. O estágio mais demorado é o gRPC: ~20 min com 8 vCPU.
