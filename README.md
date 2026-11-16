# Olá mundo com as dependências do transcritor em C++

Baixa e compila as mesmas dependências do transcritor em C++, nas mesmas versões: CUDA 12.8, TensorRT 10.16.1,
torch de CPU, os pacotes da Wolfi e o gRPC 1.85.1. Serve para testar se a CI consegue baixar tudo.

    docker build -t ola-transcritor-cpp .
    docker run --rm ola-transcritor-cpp                 # sem GPU
    docker run --rm --gpus all ola-transcritor-cpp      # com GPU: roda também um kernel e o TensorRT

O programa (`ola.cpp` + `ola.cu`) roda uma vez no fim do build: chama um servidor gRPC e mostra a versão de
cada lib. Termina com `tudo ok`.

A CI precisa alcançar:

- `registry-1.docker.io` e `cgr.dev` (as imagens base);
- `deb.debian.org` e `developer.download.nvidia.com` (apt);
- `pypi.nvidia.com`, `download.pytorch.org`, `pypi.org` e `files.pythonhosted.org` (pip);
- `github.com`, `codeload.github.com` e `raw.githubusercontent.com`;
- `packages.wolfi.dev` (apk).

Tudo é x86_64. A parte demorada é compilar o gRPC: ~20 min com 8 vCPU.
