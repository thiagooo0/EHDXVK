#include "dxvk_shader_cache.h"

#include "dxvk_log_util.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "../util/log/log.h"
#include "../util/util_env.h"
#include "../util/util_string.h"

namespace dxvk {

  namespace {
    constexpr uint32_t g_shaderCacheMagic   = 0x43535844; // 'DXSC'
    constexpr uint32_t g_shaderCacheVersion = 1;
  }

  DxvkShaderCache::DxvkShaderCache(const DxvkOptions& options, std::wstring cacheFile)
  : m_enabled(options.enableShaderCache),
    m_cacheFileName(getCacheFileName(cacheFile)) {
    if (!m_enabled) {
      Logger::info(log::ehang("Shader cache disabled, skipping cache file initialization"));
      return;
    }

    Logger::info(log::ehang("Initializing shader cache at ", str::fromws(m_cacheFileName.c_str())));
    initializeCacheFile();
  }


  DxvkShaderCache::~DxvkShaderCache() {
    stopWriter();
  }


  void DxvkShaderCache::storeShader(const DxvkShaderKey& key, const Rc<DxvkShader>& shader) {
    if (!m_enabled) {
      Logger::info(log::ehang("Shader cache disabled, ignoring shader store request"));
      return;
    }

    if (shader == nullptr) {
      Logger::warn(log::ehang("Ignoring null shader store request"));
      return;
    }

    DxvkShaderKey nullKey;
    if (key.eq(nullKey)) {
      Logger::warn(log::ehang("Ignoring shader with null cache key"));
      return;
    }

    std::string keyString = key.toString();

    {
      std::lock_guard<dxvk::mutex> lock(m_knownMutex);
      if (!m_knownShaders.insert(key).second) {
        Logger::info(log::ehang("Shader ", keyString, " already stored in cache, skipping duplicate"));
        return;
      }
    }

    Logger::info(log::ehang("Capturing shader ", keyString, " for cache persistence"));

    SerializedEntry entry = captureShader(key, shader);
    if (entry.info.codeSize == 0 || entry.spirvCode.empty()) {
      Logger::warn(log::ehang("Shader serialization produced empty SPIR-V payload"));
      return;
    }

    std::vector<char> payload = serializeEntry(entry);
    if (payload.empty()) {
      Logger::warn(log::ehang("Failed to serialize shader cache entry"));
      return;
    }

    {
      std::lock_guard<dxvk::mutex> lock(m_writerMutex);
      if (m_stopWriter.load()) {
        Logger::warn(log::ehang("Shader cache writer stopped, dropping payload"));
        return;
      }

      m_writerQueue.push(std::move(payload));
    }

    m_writerCond.notify_one();
    createWriter();
    Logger::info(log::ehang("Queued shader ", keyString, " for cache persistence"));
  }


  bool DxvkShaderCache::loadShaders(const LoadCallback& callback) {
    if (!m_enabled) {
      Logger::info(log::ehang("Shader cache disabled, skipping shader load"));
      return false;
    }

    Logger::info(log::ehang("Opening shader cache file ", str::fromws(m_cacheFileName.c_str())));

    std::ifstream stream(m_cacheFileName.c_str(), std::ios::binary);
    if (!stream) {
      Logger::info(log::ehang("Shader cache file not found, skipping load"));
      return false;
    }

    struct Header {
      uint32_t magic;
      uint32_t version;
    } header = { };

    if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header))) {
      Logger::warn(log::ehang("Failed to read shader cache header"));
      return false;
    }

    if (header.magic != g_shaderCacheMagic || header.version != g_shaderCacheVersion) {
      Logger::warn(log::ehang("Ignoring incompatible shader cache"));
      return false;
    }

    bool success = true;
    size_t loadedShaders = 0;

    while (true) {
      SerializedEntry entry;

      if (!readEntry(stream, entry)) {
        if (stream.eof())
          break;

        Logger::warn(log::ehang("Failed to read shader cache entry, discarding remainder"));
        stream.clear();
        stream.seekg(0, std::ios::end);
        success = false;
        break;
      }

      std::string keyString = entry.key.toString();
      Logger::info(log::ehang("Parsed shader cache entry for ", keyString));

      {
        std::lock_guard<dxvk::mutex> lock(m_knownMutex);
        if (!m_knownShaders.insert(entry.key).second) {
          Logger::info(log::ehang("Shader ", keyString, " already restored during this run, skipping duplicate entry"));
          continue;
        }
      }

      Rc<DxvkShader> shader = recreateShader(entry);
      if (shader == nullptr) {
        Logger::warn(log::ehang("Failed to recreate shader for ", keyString, " from cache entry"));
        success = false;
        continue;
      }

      loadedShaders += 1;
      Logger::info(log::ehang("Restored shader ", keyString, " from cache entry"));

      if (callback)
        callback(shader);
    }

    Logger::info(log::ehang("Loaded ", loadedShaders, " shaders from cache"));
    return success;
  }


  void DxvkShaderCache::stopWriter() {
    if (!m_enabled)
      return;

    bool expected = false;
    if (!m_stopWriter.compare_exchange_strong(expected, true)) {
      Logger::info(log::ehang("Shader cache writer already stopped"));
      return;
    }

    {
      std::lock_guard<dxvk::mutex> lock(m_writerMutex);
    }

    m_writerCond.notify_all();

    if (m_writerThread.joinable()) {
      m_writerThread.join();
      Logger::info(log::ehang("Shader cache writer thread stopped"));
    }
  }


  void DxvkShaderCache::initializeCacheFile() {
    struct Header {
      uint32_t magic = g_shaderCacheMagic;
      uint32_t version = g_shaderCacheVersion;
    } header;

    std::ifstream input(m_cacheFileName.c_str(), std::ios::binary);
    bool createFile = true;

    if (input) {
      Header existing = { };
      if (input.read(reinterpret_cast<char*>(&existing), sizeof(existing))) {
        if (existing.magic == g_shaderCacheMagic && existing.version == g_shaderCacheVersion)
          createFile = false;
      }
    }

    if (!createFile) {
      Logger::info(log::ehang("Shader cache file already initialized"));
      return;
    }

    std::string dir = getCacheDir();

    std::ofstream output(m_cacheFileName.c_str(), std::ios::binary | std::ios::trunc);

    if (!output && !dir.empty()) {
      if (env::createDirectory(dir))
        output = std::ofstream(m_cacheFileName.c_str(), std::ios::binary | std::ios::trunc);
    }

    if (!output) {
      Logger::err(log::ehang("Failed to create shader cache file"));
      m_enabled = false;
      return;
    }

    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    Logger::info(log::ehang("Shader cache file header written"));
  }


  void DxvkShaderCache::createWriter() {
    if (!m_writerThread.joinable()) {
      Logger::info(log::ehang("Starting shader cache writer thread"));
      m_writerThread = dxvk::thread([this] { writerFunc(); });
    }
  }


  void DxvkShaderCache::writerFunc() {
    std::ofstream stream;
    Logger::info(log::ehang("Shader cache writer thread running"));

    while (true) {
      std::vector<char> payload;

      {
        std::unique_lock<dxvk::mutex> lock(m_writerMutex);
        m_writerCond.wait(lock, [this] {
          return m_stopWriter.load() || !m_writerQueue.empty();
        });

        if (m_writerQueue.empty()) {
          if (m_stopWriter.load())
            break;
          else
            continue;
        }

        payload = std::move(m_writerQueue.front());
        m_writerQueue.pop();
      }

      if (!stream.is_open()) {
        stream.open(m_cacheFileName.c_str(), std::ios::binary | std::ios::app);

        if (!stream) {
          Logger::err(log::ehang("Failed to open shader cache file for writing"));
          return;
        }
      }

      stream.write(payload.data(), payload.size());
      stream.flush();
      Logger::info(log::ehang("Wrote shader cache payload to disk"));
    }

    Logger::info(log::ehang("Shader cache writer thread exiting"));
  }


  std::wstring DxvkShaderCache::getCacheFileName(const std::wstring& overridePath) const {
    if (!overridePath.empty())
      return overridePath;

    std::string path = getCacheDir();

    if (!path.empty() && path.back() != env::PlatformDirSlash)
      path += env::PlatformDirSlash;

    std::string exeName = env::getExeBaseName();
    path += exeName + ".dxvk-shaders";

    return str::tows(path.c_str());
  }


  std::string DxvkShaderCache::getCacheDir() const {
    std::string dir = env::getEnvVar("DXVK_SHADER_CACHE_PATH");

    if (!dir.empty())
      return dir;

    return env::getEnvVar("DXVK_STATE_CACHE_PATH");
  }


  std::vector<char> DxvkShaderCache::serializeEntry(const SerializedEntry& entry) const {
    size_t size = sizeof(entry.key) + sizeof(entry.info)
                + entry.resourceSlots.size() * sizeof(DxvkResourceSlot)
                + entry.uniformData.size()
                + entry.spirvCode.size() * sizeof(uint32_t);

    std::vector<char> data(size);
    char* ptr = data.data();

    std::memcpy(ptr, &entry.key, sizeof(entry.key));
    ptr += sizeof(entry.key);

    std::memcpy(ptr, &entry.info, sizeof(entry.info));
    ptr += sizeof(entry.info);

    if (!entry.resourceSlots.empty()) {
      std::memcpy(ptr, entry.resourceSlots.data(), entry.resourceSlots.size() * sizeof(DxvkResourceSlot));
      ptr += entry.resourceSlots.size() * sizeof(DxvkResourceSlot);
    }

    if (!entry.uniformData.empty()) {
      std::memcpy(ptr, entry.uniformData.data(), entry.uniformData.size());
      ptr += entry.uniformData.size();
    }

    if (!entry.spirvCode.empty()) {
      std::memcpy(ptr, entry.spirvCode.data(), entry.spirvCode.size() * sizeof(uint32_t));
    }

    return data;
  }


  DxvkShaderCache::SerializedEntry DxvkShaderCache::captureShader(const DxvkShaderKey& key, const Rc<DxvkShader>& shader) const {
    SerializedEntry entry;
    entry.key = key;

    const DxvkShaderCreateInfo& info = shader->info();
    entry.info.stage              = static_cast<uint32_t>(info.stage);
    entry.info.inputMask          = info.inputMask;
    entry.info.outputMask         = info.outputMask;
    entry.info.pushConstOffset    = info.pushConstOffset;
    entry.info.pushConstSize      = info.pushConstSize;
    entry.info.xfbRasterizedStream = info.xfbRasterizedStream;
    std::memcpy(entry.info.xfbStrides, info.xfbStrides, sizeof(entry.info.xfbStrides));
    entry.info.resourceSlotCount  = info.resourceSlotCount;
    entry.info.uniformSize        = info.uniformSize;
    entry.info.codeSize           = 0;

    if (info.resourceSlotCount && info.resourceSlots) {
      entry.resourceSlots.resize(info.resourceSlotCount);
      std::memcpy(entry.resourceSlots.data(), info.resourceSlots,
        info.resourceSlotCount * sizeof(DxvkResourceSlot));
    }

    if (info.uniformSize && info.uniformData) {
      entry.uniformData.resize(info.uniformSize);
      std::memcpy(entry.uniformData.data(), info.uniformData, info.uniformSize);
    }

    std::stringstream codeStream(std::ios::in | std::ios::out | std::ios::binary);
    shader->dump(codeStream);
    std::string blob = codeStream.str();

    if (!blob.empty()) {
      if (blob.size() % sizeof(uint32_t) != 0 || blob.size() > std::numeric_limits<uint32_t>::max()) {
        Logger::warn(log::ehang("Invalid SPIR-V blob when serializing shader"));
      } else {
        entry.info.codeSize = static_cast<uint32_t>(blob.size());
        entry.spirvCode.resize(blob.size() / sizeof(uint32_t));
        std::memcpy(entry.spirvCode.data(), blob.data(), blob.size());
      }
    }

    return entry;
  }


  bool DxvkShaderCache::readEntry(std::istream& stream, SerializedEntry& entry) const {
    if (!stream.read(reinterpret_cast<char*>(&entry.key), sizeof(entry.key))) {
      Logger::warn(log::ehang("Failed to read shader cache key"));
      return false;
    }

    if (!stream.read(reinterpret_cast<char*>(&entry.info), sizeof(entry.info))) {
      Logger::warn(log::ehang("Failed to read shader cache entry info"));
      return false;
    }

    if (entry.info.resourceSlotCount) {
      entry.resourceSlots.resize(entry.info.resourceSlotCount);
      if (!stream.read(reinterpret_cast<char*>(entry.resourceSlots.data()), entry.resourceSlots.size() * sizeof(DxvkResourceSlot))) {
        Logger::warn(log::ehang("Failed to read shader cache resource slots"));
        return false;
      }
    } else {
      entry.resourceSlots.clear();
    }

    if (entry.info.uniformSize) {
      entry.uniformData.resize(entry.info.uniformSize);
      if (!stream.read(entry.uniformData.data(), entry.uniformData.size())) {
        Logger::warn(log::ehang("Failed to read shader cache uniform data"));
        return false;
      }
    } else {
      entry.uniformData.clear();
    }

    if (entry.info.codeSize) {
      if (entry.info.codeSize % sizeof(uint32_t)) {
        Logger::warn(log::ehang("Shader cache entry code size is not aligned"));
        return false;
      }

      entry.spirvCode.resize(entry.info.codeSize / sizeof(uint32_t));
      if (!stream.read(reinterpret_cast<char*>(entry.spirvCode.data()), entry.info.codeSize)) {
        Logger::warn(log::ehang("Failed to read shader cache SPIR-V blob"));
        return false;
      }
    } else {
      entry.spirvCode.clear();
    }

    return true;
  }


  Rc<DxvkShader> DxvkShaderCache::recreateShader(const SerializedEntry& entry) const {
    if (entry.spirvCode.empty()) {
      Logger::warn(log::ehang("Shader cache entry missing SPIR-V code"));
      return nullptr;
    }

    DxvkShaderCreateInfo info = { };
    info.stage            = static_cast<VkShaderStageFlagBits>(entry.info.stage);
    info.resourceSlotCount = entry.info.resourceSlotCount;
    info.resourceSlots     = entry.resourceSlots.empty() ? nullptr : entry.resourceSlots.data();
    info.inputMask         = entry.info.inputMask;
    info.outputMask        = entry.info.outputMask;
    info.pushConstOffset   = entry.info.pushConstOffset;
    info.pushConstSize     = entry.info.pushConstSize;
    info.uniformSize       = entry.info.uniformSize;
    info.uniformData       = entry.uniformData.empty() ? nullptr : entry.uniformData.data();
    info.xfbRasterizedStream = entry.info.xfbRasterizedStream;
    std::memcpy(info.xfbStrides, entry.info.xfbStrides, sizeof(info.xfbStrides));

    SpirvCodeBuffer code(entry.spirvCode.size(), entry.spirvCode.data());
    Rc<DxvkShader> shader = new DxvkShader(info, std::move(code));
    shader->setShaderKey(entry.key);
    return shader;
  }

}
