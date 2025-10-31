#pragma once

#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dxvk_hash.h"
#include "dxvk_shader.h"
#include "dxvk_shader_key.h"

#include "../util/util_string.h"
#include "../util/thread.h"

namespace dxvk {

  class DxvkDevice;

  class DxvkShaderCache {

  public:

    explicit DxvkShaderCache(DxvkDevice* device);
    ~DxvkShaderCache();

    void initialize();

    void onShaderCreated(const Rc<DxvkShader>& shader);

  private:

    DxvkDevice* m_device;
    bool m_enabled = false;
    bool m_streamValid = false;
    dxvk::mutex m_mutex;
    std::ofstream m_stream;
    std::unordered_set<DxvkShaderKey, DxvkHash, DxvkEq> m_knownShaders;
    std::unordered_map<DxvkShaderKey, Rc<DxvkShader>, DxvkHash, DxvkEq> m_shaderMap;
    std::vector<Rc<DxvkShader>> m_cachedShaders;

    bool loadCacheFile();
    void openStream(bool recreate);
    void writeShaderEntry(const Rc<DxvkShader>& shader,
      const SpirvCodeBuffer& rawCode);
    bool serializeShaderEntry(std::ostream& stream,
      const Rc<DxvkShader>& shader,
      const SpirvCodeBuffer& rawCode);
    std::string getCacheDir() const;
    str::path_string getCacheFileName() const;

    void registerCachedShader(const Rc<DxvkShader>& shader);

  public:

    Rc<DxvkShader> findShader(const DxvkShaderKey& key);
  };

}
