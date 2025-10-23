#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "dxvk_hash.h"
#include "dxvk_options.h"
#include "dxvk_shader.h"
#include "dxvk_shader_key.h"

#include "../util/thread.h"

namespace dxvk {

  class DxvkShaderCache : public RcObject {

  public:

    using LoadCallback = std::function<void(const Rc<DxvkShader>&)>;

    DxvkShaderCache(const DxvkOptions& options, std::wstring cacheFile = std::wstring());
    ~DxvkShaderCache();

    bool isEnabled() const {
      return m_enabled;
    }

    void storeShader(const DxvkShaderKey& key, const Rc<DxvkShader>& shader);

    bool loadShaders(const LoadCallback& callback);

    void stopWriter();

  private:

    struct EntryInfo {
      uint32_t stage;
      uint32_t inputMask;
      uint32_t outputMask;
      uint32_t pushConstOffset;
      uint32_t pushConstSize;
      int32_t  xfbRasterizedStream;
      uint32_t xfbStrides[MaxNumXfbBuffers];
      uint32_t resourceSlotCount;
      uint32_t uniformSize;
      uint32_t codeSize;
    };

    struct SerializedEntry {
      DxvkShaderKey key;
      EntryInfo     info;
      std::vector<DxvkResourceSlot> resourceSlots;
      std::vector<char>             uniformData;
      std::vector<uint32_t>         spirvCode;
    };

    bool                      m_enabled;
    std::wstring              m_cacheFileName;

    std::atomic<bool>         m_stopWriter = { false };
    dxvk::mutex               m_writerMutex;
    dxvk::condition_variable  m_writerCond;
    std::queue<std::vector<char>> m_writerQueue;
    dxvk::thread              m_writerThread;

    dxvk::mutex               m_knownMutex;
    std::unordered_set<DxvkShaderKey, DxvkHash, DxvkEq> m_knownShaders;

    void initializeCacheFile();

    void createWriter();
    void writerFunc();

    std::wstring getCacheFileName(const std::wstring& overridePath) const;
    std::string  getCacheDir() const;

    std::vector<char> serializeEntry(const SerializedEntry& entry) const;
    SerializedEntry   captureShader(const DxvkShaderKey& key, const Rc<DxvkShader>& shader) const;
    bool              readEntry(std::istream& stream, SerializedEntry& entry) const;
    Rc<DxvkShader>    recreateShader(const SerializedEntry& entry) const;
  };

}
