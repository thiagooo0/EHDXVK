#include "dxvk_shader_cache.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "dxvk_device.h"
#include "dxvk_pipelayout.h"

#include "../util/log/log.h"
#include "../util/sha1/sha1_util.h"
#include "../util/util_env.h"

namespace dxvk {

  namespace {

    constexpr uint32_t DxvkShaderCacheMagic        = 0x43535844; /* DXSC */
    constexpr uint32_t DxvkShaderCacheVersion      = 2;
    constexpr uint32_t DxvkShaderCacheEntryMagic   = 0x45484353; /* SCHE */

    bool isCacheableStage(VkShaderStageFlagBits stage) {
      switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT:
        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
        case VK_SHADER_STAGE_GEOMETRY_BIT:
        case VK_SHADER_STAGE_FRAGMENT_BIT:
          return true;
        default:
          return false;
      }
    }

    bool isKnownStageValue(uint32_t stage) {
      switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT:
        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
        case VK_SHADER_STAGE_GEOMETRY_BIT:
        case VK_SHADER_STAGE_FRAGMENT_BIT:
          return true;
        default:
          return false;
      }
    }

    bool resyncToNextEntry(std::istream& stream, std::streampos startPos) {
      stream.clear();

      if (startPos != std::streampos(-1))
        stream.seekg(startPos);

      std::streampos resumePos = stream.tellg();

      if (resumePos == std::streampos(-1))
        return false;

      uint32_t window = 0;
      size_t bytesRead = 0;
      size_t filledBytes = 0;

      while (true) {
        char byte = 0;

        if (!stream.get(byte))
          break;

        window = (window >> 8) | (uint32_t(static_cast<uint8_t>(byte)) << 24);
        bytesRead += 1;

        if (filledBytes < sizeof(uint32_t))
          filledBytes += 1;

        if (filledBytes == sizeof(uint32_t) && window == DxvkShaderCacheEntryMagic) {
          std::streampos currentPos = stream.tellg();
          std::streamoff rewind = sizeof(uint32_t);
          stream.seekg(currentPos - rewind);
          stream.clear();

          std::streamoff syncPos = stream.tellg();

          Logger::ehang(str::format(
            "Resynchronized shader cache at offset ",
            static_cast<long long>(syncPos),
            " after skipping ",
            bytesRead - sizeof(uint32_t),
            " bytes"));
          return true;
        }
      }

      stream.clear();
      stream.seekg(0, std::ios_base::end);
      Logger::ehang("Failed to resynchronize shader cache; reached end of file");
      return false;
    }

    struct DxvkShaderCacheHeader {
      uint32_t magic;
      uint32_t version;
    };

    struct DxvkShaderCacheEntryHeader {
      uint32_t stage;
      uint32_t bindingCount;
      uint32_t uniformSize;
      uint32_t codeSize;
      uint32_t inputMask;
      uint32_t outputMask;
      uint32_t flatShadingInputs;
      uint32_t pushConstOffset;
      uint32_t pushConstSize;
      int32_t  xfbRasterizedStream;
      uint32_t patchVertexCount;
      uint32_t xfbStrides[MaxNumXfbBuffers];
      uint32_t outputTopology;
    };

    constexpr uint32_t DxvkShaderCacheEntryHeaderSize = sizeof(DxvkShaderCacheEntryHeader);
    constexpr uint32_t DxvkShaderCacheDigestSize = sizeof(Sha1Digest);

    void writeDigest(std::ostream& stream, const Sha1Hash& hash) {
      Sha1Digest digest;

      for (uint32_t i = 0; i < digest.size() / 4; i++) {
        uint32_t dword = hash.dword(i);
        digest[4 * i + 0] = uint8_t((dword >> 0)  & 0xff);
        digest[4 * i + 1] = uint8_t((dword >> 8)  & 0xff);
        digest[4 * i + 2] = uint8_t((dword >> 16) & 0xff);
        digest[4 * i + 3] = uint8_t((dword >> 24) & 0xff);
      }

      stream.write(reinterpret_cast<const char*>(digest.data()), digest.size());
    }

    Sha1Hash makeHash(const Sha1Digest& digest) {
      return Sha1Hash(digest);
    }

  }

  DxvkShaderCache::DxvkShaderCache(DxvkDevice* device)
  : m_device(device) {

  }


  DxvkShaderCache::~DxvkShaderCache() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    if (m_stream.is_open())
      m_stream.close();
  }


  void DxvkShaderCache::initialize() {
    std::string useShaderCache = env::getEnvVar("DXVK_SHADER_CACHE");

    m_enabled = useShaderCache != "0" && useShaderCache != "disable"
             && m_device->config().enableShaderCache;

    if (!m_enabled) {
      Logger::ehang("Shader cache disabled");
      return;
    }

    Logger::ehang("Initializing shader cache");

    bool recreate = useShaderCache == "reset";
    bool loadedFromDisk = false;

    if (!recreate) {
      loadedFromDisk = loadCacheFile();

      if (!m_cachedShaders.empty()) {
        for (const auto& shader : m_cachedShaders)
          registerCachedShader(shader);

        m_device->prewarmCachedPipelines();
        m_cachedShaders.clear();
      }

      if (!loadedFromDisk)
        recreate = true;
    }

    openStream(recreate);
  }


  void DxvkShaderCache::onShaderCreated(const Rc<DxvkShader>& shader) {
    if (!m_enabled)
      return;

    if (!isCacheableStage(shader->info().stage))
      return;

    DxvkShaderKey key = shader->getShaderKey();
    DxvkShaderKey nullKey;

    SpirvCodeBuffer rawCode = shader->getRawCode();

    if (!rawCode.size()) {
      Logger::ehang(str::format(
        "Skipping shader without SPIR-V payload for stage ",
        uint32_t(shader->info().stage)));
      return;
    }

    if (key.eq(nullKey)) {
      key = DxvkShaderKey(shader->info().stage,
        Sha1Hash::compute(rawCode.data(), rawCode.size()));
      shader->setShaderKey(key);

      Logger::ehang(str::format(
        "Generated implicit shader key for stage ",
        uint32_t(shader->info().stage), " hash ", key.toString()));
    }

    std::lock_guard<dxvk::mutex> lock(m_mutex);

    auto insertResult = m_knownShaders.insert(key);

    m_shaderMap[key] = shader;

    if (!insertResult.second)
      return;

    writeShaderEntry(shader, rawCode);
  }


  bool DxvkShaderCache::loadCacheFile() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    m_knownShaders.clear();
    m_shaderMap.clear();
    m_cachedShaders.clear();
    auto fileName = getCacheFileName();

    std::ifstream stream(fileName.c_str(), std::ios_base::binary);

    if (!stream) {
      Logger::ehang("Shader cache file not found, a new one will be created");
      return false;
    }

    DxvkShaderCacheHeader header = { };

    if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header))) {
      Logger::ehang("Failed to read shader cache header");
      return false;
    }

    if (header.magic != DxvkShaderCacheMagic) {
      Logger::ehang("Invalid shader cache magic, recreating cache");
      return false;
    }

    if (header.version != DxvkShaderCacheVersion) {
      Logger::ehang("Incompatible shader cache version, recreating cache");
      return false;
    }

    size_t shaderCount = 0;

    bool encounteredCorruption = false;
    bool sawEntryMagic = false;

    while (stream.good()) {
      std::streampos entryOffset = stream.tellg();

      uint32_t firstWord = 0;
      stream.read(reinterpret_cast<char*>(&firstWord), sizeof(firstWord));

      if (!stream) {
        if (!stream.eof())
          encounteredCorruption = true;
        break;
      }

      uint32_t stage = 0;
      bool entryHasMagic = false;

      if (firstWord == DxvkShaderCacheEntryMagic) {
        entryHasMagic = true;
        sawEntryMagic = true;

        stream.read(reinterpret_cast<char*>(&stage), sizeof(stage));

        if (!stream) {
          Logger::ehang("Unexpected end of shader cache while reading entry stage");
          encounteredCorruption = true;

          if (resyncToNextEntry(stream, entryOffset + std::streamoff(sizeof(uint32_t))))
            continue;
          break;
        }
      } else {
        stage = firstWord;

        if (!isKnownStageValue(stage) && sawEntryMagic) {
          Logger::ehang(str::format(
            "Invalid shader cache entry: unexpected stage value ", stage));
          encounteredCorruption = true;

          if (resyncToNextEntry(stream, entryOffset + std::streamoff(sizeof(uint32_t))))
            continue;
          break;
        }
      }

      std::streampos sizePos = stream.tellg();

      uint32_t entrySize = 0;
      stream.read(reinterpret_cast<char*>(&entrySize), sizeof(entrySize));

      if (!stream) {
        Logger::ehang("Unexpected end of shader cache while reading entry size");
        encounteredCorruption = true;
        if ((entryHasMagic || sawEntryMagic) &&
            resyncToNextEntry(stream, sizePos))
          continue;
        break;
      }

      std::streampos payloadPos = stream.tellg();

      if (entrySize < DxvkShaderCacheDigestSize + DxvkShaderCacheEntryHeaderSize) {
        Logger::ehang(str::format(
          "Invalid shader cache entry: entry size ", entrySize,
          " smaller than required header"));
        encounteredCorruption = true;

        if (entryHasMagic || sawEntryMagic) {
          if (resyncToNextEntry(stream, payloadPos))
            continue;
          break;
        }
        continue;
      }

      std::vector<char> entryData(entrySize);

      if (!stream.read(entryData.data(), entryData.size())) {
        Logger::ehang("Unexpected end of shader cache while reading entry payload");
        encounteredCorruption = true;
        if ((entryHasMagic || sawEntryMagic) &&
            resyncToNextEntry(stream, payloadPos))
          continue;
        break;
      }

      size_t offset = 0;

      Sha1Digest digest = { };
      std::memcpy(digest.data(), entryData.data(), DxvkShaderCacheDigestSize);
      offset += DxvkShaderCacheDigestSize;

      DxvkShaderCacheEntryHeader entry = { };
      std::memcpy(&entry, entryData.data() + offset, DxvkShaderCacheEntryHeaderSize);
      offset += DxvkShaderCacheEntryHeaderSize;

      Logger::ehang(str::format(
        "Reading shader entry stage ", stage,
        " codeSize=", entry.codeSize,
        " entrySize=", entrySize));

      if (entry.stage != stage) {
        Logger::ehang("Shader cache entry stage mismatch");
        encounteredCorruption = true;
        if (entryHasMagic || sawEntryMagic) {
          if (resyncToNextEntry(stream, payloadPos))
            continue;
          break;
        }
        continue;
      }

      if (entry.codeSize & 3) {
        Logger::ehang("Invalid shader cache entry: corrupted SPIR-V size");
        encounteredCorruption = true;
        if (entryHasMagic || sawEntryMagic) {
          if (resyncToNextEntry(stream, payloadPos))
            continue;
          break;
        }
        continue;
      }

      size_t bindingSize = size_t(entry.bindingCount) * sizeof(DxvkBindingInfo);

      if (offset + bindingSize > entryData.size()) {
        Logger::ehang("Invalid shader cache entry: truncated binding data");
        encounteredCorruption = true;
        if (entryHasMagic || sawEntryMagic) {
          if (resyncToNextEntry(stream, payloadPos))
            continue;
          break;
        }
        continue;
      }

      std::vector<DxvkBindingInfo> bindings(entry.bindingCount);

      if (bindingSize)
        std::memcpy(bindings.data(), entryData.data() + offset, bindingSize);

      offset += bindingSize;

      size_t uniformSize = entry.uniformSize;

      if (offset + uniformSize > entryData.size()) {
        Logger::ehang("Invalid shader cache entry: truncated uniform data");
        encounteredCorruption = true;
        if (entryHasMagic || sawEntryMagic) {
          if (resyncToNextEntry(stream, payloadPos))
            continue;
          break;
        }
        continue;
      }

      std::vector<char> uniformData(uniformSize);

      if (uniformSize)
        std::memcpy(uniformData.data(), entryData.data() + offset, uniformSize);

      offset += uniformSize;

      if (entry.codeSize == 0) {
        Logger::ehang("Skipping shader cache entry without SPIR-V payload");
        continue;
      }

      if (offset + entry.codeSize > entryData.size()) {
        Logger::ehang("Invalid shader cache entry: truncated SPIR-V payload");
        encounteredCorruption = true;
        if (entryHasMagic || sawEntryMagic) {
          if (resyncToNextEntry(stream, payloadPos))
            continue;
          break;
        }
        continue;
      }

      std::vector<uint32_t> spirv(entry.codeSize / sizeof(uint32_t));

      if (!spirv.empty())
        std::memcpy(spirv.data(), entryData.data() + offset, entry.codeSize);

      offset += entry.codeSize;

      if (offset != entryData.size()) {
        Logger::ehang(str::format(
          "Shader cache entry contains ", entryData.size() - offset,
          " bytes of unexpected padding"));
        encounteredCorruption = true;
        if (entryHasMagic || sawEntryMagic) {
          if (resyncToNextEntry(stream, payloadPos))
            continue;
          break;
        }
        continue;
      }

      DxvkShaderCreateInfo info = { };
      info.stage             = VkShaderStageFlagBits(entry.stage);
      info.bindingCount      = entry.bindingCount;
      info.bindings          = bindings.data();
      info.inputMask         = entry.inputMask;
      info.outputMask        = entry.outputMask;
      info.flatShadingInputs = entry.flatShadingInputs;
      info.pushConstOffset   = entry.pushConstOffset;
      info.pushConstSize     = entry.pushConstSize;
      info.uniformSize       = entry.uniformSize;
      info.uniformData       = uniformData.empty() ? nullptr : uniformData.data();
      info.xfbRasterizedStream = entry.xfbRasterizedStream;
      info.patchVertexCount    = entry.patchVertexCount;
      std::memcpy(info.xfbStrides, entry.xfbStrides, sizeof(entry.xfbStrides));
      info.outputTopology      = VkPrimitiveTopology(entry.outputTopology);

      SpirvCodeBuffer codeBuffer(static_cast<uint32_t>(spirv.size()), spirv.data());

      Rc<DxvkShader> shader = new DxvkShader(info, std::move(codeBuffer));
      shader->setShaderKey(DxvkShaderKey(info.stage, makeHash(digest)));

      if (isCacheableStage(info.stage)) {
        DxvkShaderKey shaderKey = shader->getShaderKey();
        m_knownShaders.insert(shaderKey);
        m_shaderMap[shaderKey] = shader;

        m_cachedShaders.push_back(shader);
        shaderCount += 1;
      } else {
        Logger::ehang(str::format("Skipping non-pre-raster/fragment shader stage ", uint32_t(info.stage)));
      }
    }

    if (encounteredCorruption)
      Logger::ehang("Shader cache contained corrupted entries that were skipped");

    Logger::ehang(str::format("Loaded ", shaderCount, " shaders from cache"));
    return shaderCount > 0 || !encounteredCorruption;
  }


  void DxvkShaderCache::openStream(bool recreate) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    auto fileName = getCacheFileName();

    if (fileName.empty()) {
      Logger::ehang("Cannot determine shader cache path, disabling cache");
      m_enabled = false;
      m_streamValid = false;
      return;
    }

    std::ios_base::openmode mode = std::ios_base::binary;
    mode |= recreate ? std::ios_base::trunc : std::ios_base::app;

    m_stream.close();
    m_stream.clear();

    m_stream = std::ofstream(fileName.c_str(), mode);

    if (!m_stream && env::createDirectory(getCacheDir())) {
      m_stream = std::ofstream(fileName.c_str(), mode);
    }

    if (!m_stream) {
      Logger::ehang("Failed to open shader cache for writing, disabling cache");
      m_enabled = false;
      m_streamValid = false;
      return;
    }

    if (recreate) {
      DxvkShaderCacheHeader header = { DxvkShaderCacheMagic, DxvkShaderCacheVersion };
      m_stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
      m_stream.flush();
    } else if (m_stream.tellp() == std::streampos(0)) {
      DxvkShaderCacheHeader header = { DxvkShaderCacheMagic, DxvkShaderCacheVersion };
      m_stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
      m_stream.flush();
    }

    m_streamValid = true;
  }


  void DxvkShaderCache::writeShaderEntry(const Rc<DxvkShader>& shader,
      const SpirvCodeBuffer& rawCode) {
    if (!m_streamValid)
      return;

    if (!serializeShaderEntry(m_stream, shader, rawCode))
      return;

    m_stream.flush();

    Logger::ehang(str::format("Cached shader ", shader->getShaderKey().toString()));
  }


  bool DxvkShaderCache::serializeShaderEntry(std::ostream& stream,
      const Rc<DxvkShader>& shader,
      const SpirvCodeBuffer& rawCode) {
    const DxvkShaderCreateInfo& info = shader->info();

    if (!isCacheableStage(info.stage))
      return false;

    if (!rawCode.size()) {
      Logger::ehang(str::format(
        "Skipping shader ", shader->debugName(),
        " because SPIR-V payload is empty"));
      return false;
    }

    if (rawCode.size() & 3) {
      Logger::ehang(str::format(
        "Skipping shader ", shader->debugName(),
        " because SPIR-V payload has invalid size ", rawCode.size()));
      return false;
    }

    std::vector<DxvkBindingInfo> bindings;
    bindings.reserve(info.bindingCount);

    const DxvkBindingLayout& layout = shader->getBindings();

    for (uint32_t set = 0; set < DxvkDescriptorSets::SetCount; set++) {
      uint32_t count = layout.getBindingCount(set);

      for (uint32_t i = 0; i < count; i++)
        bindings.push_back(layout.getBinding(set, i));
    }

    std::vector<char> uniformData;

    if (info.uniformSize && info.uniformData)
      uniformData.assign(info.uniformData, info.uniformData + info.uniformSize);

    std::vector<uint32_t> spirv(rawCode.dwords());

    if (!spirv.empty())
      std::memcpy(spirv.data(), rawCode.data(), rawCode.size());

    DxvkShaderCacheEntryHeader header = { };
    header.stage             = static_cast<uint32_t>(info.stage);
    header.bindingCount      = static_cast<uint32_t>(bindings.size());
    header.uniformSize       = info.uniformSize;
    header.codeSize          = static_cast<uint32_t>(rawCode.size());
    header.inputMask         = info.inputMask;
    header.outputMask        = info.outputMask;
    header.flatShadingInputs = info.flatShadingInputs;
    header.pushConstOffset   = info.pushConstOffset;
    header.pushConstSize     = info.pushConstSize;
    header.xfbRasterizedStream = info.xfbRasterizedStream;
    header.patchVertexCount    = info.patchVertexCount;
    std::memcpy(header.xfbStrides, info.xfbStrides, sizeof(header.xfbStrides));
    header.outputTopology      = static_cast<uint32_t>(info.outputTopology);

    size_t bindingSize = bindings.size() * sizeof(DxvkBindingInfo);
    size_t entrySize = DxvkShaderCacheDigestSize
      + DxvkShaderCacheEntryHeaderSize
      + bindingSize
      + uniformData.size()
      + header.codeSize;

    if (entrySize > std::numeric_limits<uint32_t>::max()) {
      Logger::ehang(str::format(
        "Skipping shader ", shader->debugName(),
        " because cache entry exceeds size limit"));
      return false;
    }

    DxvkShaderKey key = shader->getShaderKey();

    uint32_t stageValue = header.stage;
    uint32_t entrySizeValue = static_cast<uint32_t>(entrySize);

    Logger::ehang(str::format(
      "Writing shader entry stage ", stageValue,
      " codeSize=", header.codeSize,
      " entrySize=", entrySizeValue));

    uint32_t entryMagic = DxvkShaderCacheEntryMagic;

    stream.write(reinterpret_cast<const char*>(&entryMagic), sizeof(entryMagic));
    stream.write(reinterpret_cast<const char*>(&stageValue), sizeof(stageValue));
    stream.write(reinterpret_cast<const char*>(&entrySizeValue), sizeof(entrySizeValue));
    writeDigest(stream, key.sha1());
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));

    if (!bindings.empty())
      stream.write(reinterpret_cast<const char*>(bindings.data()), bindings.size() * sizeof(DxvkBindingInfo));

    if (!uniformData.empty())
      stream.write(uniformData.data(), uniformData.size());

    if (!spirv.empty())
      stream.write(reinterpret_cast<const char*>(spirv.data()), header.codeSize);

    if (!stream.good()) {
      Logger::ehang(str::format(
        "Failed to serialize shader cache entry for ", shader->debugName()));
      return false;
    }

    return true;
  }



  std::string DxvkShaderCache::getCacheDir() const {
    std::string path = env::getEnvVar("DXVK_SHADER_CACHE_PATH");

    if (path.empty())
      path = env::getEnvVar("DXVK_STATE_CACHE_PATH");

    return path;
  }


  str::path_string DxvkShaderCache::getCacheFileName() const {
    std::string path = getCacheDir();

    if (!path.empty() && *path.rbegin() != '/')
      path += '/';

    std::string exeName = env::getExeBaseName();
    path += exeName + ".dxvk-shaders";

    return str::topath(path.c_str());
  }


  void DxvkShaderCache::registerCachedShader(const Rc<DxvkShader>& shader) {
    Logger::ehang(str::format("Registering cached shader ", shader->debugName()));
    DxvkShaderPipelineLibrary* library = m_device->registerShaderFromCache(shader);
    m_device->prewarmShaderWithGraphics(shader, library);
  }


  Rc<DxvkShader> DxvkShaderCache::findShader(const DxvkShaderKey& key) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    auto entry = m_shaderMap.find(key);

    if (entry == m_shaderMap.end())
      return nullptr;

    return entry->second;
  }

}
