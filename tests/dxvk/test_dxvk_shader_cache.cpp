#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "../../src/dxvk/dxvk_shader_cache.h"
#include "../../src/dxvk/dxvk_shader.h"
#include "../../src/dxvk/dxvk_shader_key.h"
#include "../../src/spirv/spirv_code_buffer.h"
#include "../../src/util/config/config.h"
#include "../../src/util/log/log.h"
#include "../../src/util/util_string.h"

#include <spirv/spirv.hpp>

using namespace dxvk;

namespace dxvk {
  Logger Logger::s_instance("dxvk-shader-cache-test.log");
}

namespace {

  SpirvCodeBuffer createTestShaderCode() {
    SpirvCodeBuffer code;
    code.putHeader(0x00010000, 6);

    code.putIns(spv::OpCapability, 2);
    code.putInt32(spv::CapabilityShader);

    code.putIns(spv::OpMemoryModel, 3);
    code.putInt32(spv::AddressingModelLogical);
    code.putInt32(spv::MemoryModelGLSL450);

    uint16_t entryWordCount = 3 + code.strLen("main");
    code.putIns(spv::OpEntryPoint, entryWordCount);
    code.putInt32(spv::ExecutionModelVertex);
    code.putInt32(4);
    code.putStr("main");

    code.putIns(spv::OpTypeVoid, 2);
    code.putInt32(1);

    code.putIns(spv::OpTypeFunction, 3);
    code.putInt32(2);
    code.putInt32(1);

    code.putIns(spv::OpFunction, 5);
    code.putInt32(1);
    code.putInt32(4);
    code.putInt32(spv::FunctionControlMaskNone);
    code.putInt32(2);

    code.putIns(spv::OpLabel, 2);
    code.putInt32(5);

    code.putIns(spv::OpReturn, 1);
    code.putIns(spv::OpFunctionEnd, 1);

    return code;
  }

}

int main() {
  Config config;
  config.setOption("dxvk.shaderCache", "True");
  DxvkOptions options(config);

  const std::string cachePath = "dxvk_shader_cache_test.bin";
  std::remove(cachePath.c_str());

  SpirvCodeBuffer shaderCode = createTestShaderCode();

  DxvkShaderCreateInfo info = { };
  info.stage = VK_SHADER_STAGE_VERTEX_BIT;

  Rc<DxvkShader> shader = new DxvkShader(info, std::move(shaderCode));

  std::stringstream originalStream(std::ios::in | std::ios::out | std::ios::binary);
  shader->dump(originalStream);
  std::string originalBlob = originalStream.str();

  Sha1Hash hash = Sha1Hash::compute(originalBlob.data(), originalBlob.size());
  DxvkShaderKey key(VK_SHADER_STAGE_VERTEX_BIT, hash);
  shader->setShaderKey(key);

  DxvkShaderCache cache(options, str::tows(cachePath.c_str()));
  cache.storeShader(key, shader);
  cache.storeShader(key, shader);
  cache.stopWriter();

  DxvkShaderCache reload(options, str::tows(cachePath.c_str()));
  std::vector<Rc<DxvkShader>> restored;
  bool loadSuccess = reload.loadShaders([&restored](const Rc<DxvkShader>& loadedShader) {
    restored.push_back(loadedShader);
  });
  reload.stopWriter();

  if (!loadSuccess) {
    std::cerr << "Failed to load shader cache" << std::endl;
    return 1;
  }

  if (restored.size() != 1) {
    std::cerr << "Unexpected shader count: " << restored.size() << std::endl;
    return 1;
  }

  const Rc<DxvkShader>& restoredShader = restored[0];
  if (!restoredShader->getShaderKey().eq(key)) {
    std::cerr << "Shader key mismatch" << std::endl;
    return 1;
  }

  std::stringstream restoredStream(std::ios::in | std::ios::out | std::ios::binary);
  restoredShader->dump(restoredStream);

  if (restoredStream.str() != originalBlob) {
    std::cerr << "Shader code mismatch" << std::endl;
    return 1;
  }

  std::remove(cachePath.c_str());
  return 0;
}
