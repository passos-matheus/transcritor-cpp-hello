// Olá mundo com todas as dependências do transcritor em C++: gRPC (servidor e cliente de verdade, em
// 127.0.0.1), protobuf, abseil, OpenSSL (sha256), zlib, nlohmann-json, CUDA (cudart, cuBLAS, cuFFT e
// um kernel), TensorRT e o parser de ONNX. Sem GPU, mostra as versões e diz que não há GPU; com GPU,
// roda o kernel e cria um builder do TensorRT e um parser de ONNX. Sai com 0 se tudo carregou.
#include <cstdio>
#include <memory>
#include <string>

#include <absl/base/config.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <grpcpp/grpcpp.h>
#include <nlohmann/json.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <zlib.h>

#include "ola.grpc.pb.h"

extern "C" int ola_cuda(int* soma);

namespace {

class Servico final : public ola::Ola::Service {
  grpc::Status Diga(grpc::ServerContext*, const ola::Pedido* p, ola::Resposta* r) override {
    r->set_texto("olá, " + p->nome());
    return grpc::Status::OK;
  }
};

class Registro : public nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override {
    if (s <= Severity::kWARNING) std::printf("  [TensorRT] %s\n", m);
  }
};

std::string sha256(const std::string& s) {
  unsigned char h[EVP_MAX_MD_SIZE];
  unsigned int n = 0;
  EVP_Digest(s.data(), s.size(), h, &n, EVP_sha256(), nullptr);
  std::string hex;
  char b[3];
  for (unsigned int i = 0; i < n; i++) {
    std::snprintf(b, sizeof b, "%02x", h[i]);
    hex += b;
  }
  return hex;
}

}  // namespace

int main() {
  int falhas = 0;
  std::printf("olá mundo: as dependências do transcritor em C++\n");

  // gRPC + protobuf: um servidor numa porta livre e uma chamada a ele
  Servico servico;
  int porta = 0;
  grpc::ServerBuilder b;
  b.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &porta);
  b.RegisterService(&servico);
  std::unique_ptr<grpc::Server> servidor = b.BuildAndStart();
  if (!servidor || porta == 0) {
    std::printf("gRPC %s: o servidor NÃO subiu\n", grpc::Version().c_str());
    falhas++;
  } else {
    auto canal = grpc::CreateChannel("127.0.0.1:" + std::to_string(porta), grpc::InsecureChannelCredentials());
    auto stub = ola::Ola::NewStub(canal);
    ola::Pedido p;
    p.set_nome("mundo");
    ola::Resposta r;
    grpc::ClientContext ctx;
    grpc::Status st = stub->Diga(&ctx, p, &r);
    std::printf("gRPC %s: %s\n", grpc::Version().c_str(),
                st.ok() ? ("\"" + r.texto() + "\" pela porta " + std::to_string(porta)).c_str()
                        : ("ERRO " + st.error_message()).c_str());
    if (!st.ok() || r.texto() != "olá, mundo") falhas++;
    servidor->Shutdown();
  }

#ifdef ABSL_LTS_RELEASE_VERSION
  std::printf("abseil LTS %d.%d\n", ABSL_LTS_RELEASE_VERSION, ABSL_LTS_RELEASE_PATCH_LEVEL);
#endif
  std::string h = sha256("olá");
  std::printf("%s: sha256(\"olá\") = %s\n", OpenSSL_version(OPENSSL_VERSION), h.c_str());
  std::printf("zlib %s\n", zlibVersion());
  nlohmann::json j = nlohmann::json::parse(R"({"ola": "mundo", "n": 1})");
  std::printf("nlohmann-json %d.%d.%d: %s\n", NLOHMANN_JSON_VERSION_MAJOR, NLOHMANN_JSON_VERSION_MINOR,
              NLOHMANN_JSON_VERSION_PATCH, j.dump().c_str());

  // CUDA: as versões das libs (não precisam de GPU)
  int rt = 0, cb_maior = 0, cb_menor = 0, cb_patch = 0, cf = 0;
  cudaRuntimeGetVersion(&rt);
  cublasGetProperty(MAJOR_VERSION, &cb_maior);
  cublasGetProperty(MINOR_VERSION, &cb_menor);
  cublasGetProperty(PATCH_LEVEL, &cb_patch);
  cufftGetVersion(&cf);
  std::printf("CUDA runtime %d.%d, cuBLAS %d.%d.%d, cuFFT %d\n", rt / 1000, (rt % 1000) / 10, cb_maior, cb_menor,
              cb_patch, cf);

  // TensorRT e o parser de ONNX (também sem GPU)
  int trt = getInferLibVersion();
  std::printf("TensorRT %d.%d.%d, parser de ONNX %d\n", trt / 10000, (trt / 100) % 100, trt % 100,
              getNvOnnxParserVersion());

  int gpus = 0;
  cudaError_t e = cudaGetDeviceCount(&gpus);
  if (e != cudaSuccess || gpus == 0) {
    std::printf("sem GPU aqui (%s): o kernel e o builder do TensorRT ficam para uma máquina com GPU\n",
                e != cudaSuccess ? cudaGetErrorString(e) : "0 GPUs");
  } else {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int soma = 0;
    int k = ola_cuda(&soma);
    std::printf("GPU %s (cc %d.%d): kernel %s (soma %d, esperado 992)\n", prop.name, prop.major, prop.minor,
                k == 0 && soma == 992 ? "ok" : "FALHOU", soma);
    if (k != 0 || soma != 992) falhas++;
    Registro reg;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(reg));
    std::unique_ptr<nvinfer1::INetworkDefinition> rede(builder ? builder->createNetworkV2(0U) : nullptr);
    std::unique_ptr<nvonnxparser::IParser> parser(rede ? nvonnxparser::createParser(*rede, reg) : nullptr);
    std::printf("TensorRT na GPU: builder %s, parser de ONNX %s\n", builder ? "ok" : "FALHOU",
                parser ? "ok" : "FALHOU");
    if (!builder || !parser) falhas++;
  }

  std::printf(falhas == 0 ? "tudo ok\n" : "%d FALHA(S)\n", falhas);
  return falhas == 0 ? 0 : 1;
}
