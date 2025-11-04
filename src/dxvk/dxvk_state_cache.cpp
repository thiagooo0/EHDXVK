#include "dxvk_device.h"
#include "dxvk_pipemanager.h"
#include "dxvk_state_cache.h"

#include "../util/log/log.h"
#include "../util/util_string.h"

#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace dxvk {

  static const Sha1Hash       g_nullHash      = Sha1Hash::compute(nullptr, 0);
  static const DxvkShaderKey  g_nullShaderKey = DxvkShaderKey();

  namespace {

    std::string describeSpecConstantState(uint32_t mask, const DxvkScInfo& scInfo) {
      std::ostringstream stream;
      stream << "mask=0x" << std::hex << mask << std::dec;

      bool hasValues = false;

      for (auto constantId : bit::BitMask(mask & ((1u << MaxNumSpecConstants) - 1u))) {
        uint32_t value = scInfo.specConstants[constantId];

        stream << (hasValues ? ", " : " values=[");
        stream << constantId << "=0x" << std::hex << value << std::dec;
        hasValues = true;
      }

      if (hasValues)
        stream << ']';

      return stream.str();
    }

    std::string describeComputeState(const DxvkStateCacheKey& key, uint32_t mask, const DxvkComputePipelineStateInfo& state) {
      return str::format(
        "shader=", key.cs.toString(), " ",
        describeSpecConstantState(mask, state.sc));
    }

  }


  /**
   * \brief Packed entry header
   */
  struct DxvkStateCacheEntryHeader {
    uint32_t entryType : 2;
    uint32_t stageMask : 6;
    uint32_t entrySize : 24;
  };


  /**
   * \brief Version 8 entry header
   */
  struct DxvkStateCacheEntryHeaderV8 {
    uint32_t stageMask : 8;
    uint32_t entrySize : 24;
  };


  /**
   * \brief Version 16 entry header
   */
  struct DxvkStateCacheEntryHeaderV16 {
    uint32_t entryType : 1;
    uint32_t stageMask : 5;
    uint32_t entrySize : 26;
  };

  
  /**
   * \brief State cache entry data
   *
   * Stores data for a single cache entry and
   * provides convenience methods to access it.
   */
  class DxvkStateCacheEntryData {
    constexpr static size_t MaxSize = 1024;
  public:

    size_t size() const {
      return m_size;
    }

    const char* data() const {
      return m_data;
    }

    Sha1Hash computeHash() const {
      return Sha1Hash::compute(m_data, m_size);
    }

    template<typename T>
    bool read(T& data, uint32_t version) {
      return read(data);
    }

    bool read(DxvkStateCacheKey& shaders, uint32_t version, VkShaderStageFlags stageFlags) {
      std::array<std::pair<VkShaderStageFlagBits, DxvkShaderKey*>, 6> stages = {{
        { VK_SHADER_STAGE_VERTEX_BIT,                   &shaders.vs },
        { VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,     &shaders.tcs },
        { VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,  &shaders.tes },
        { VK_SHADER_STAGE_GEOMETRY_BIT,                 &shaders.gs },
        { VK_SHADER_STAGE_FRAGMENT_BIT,                 &shaders.fs },
        { VK_SHADER_STAGE_COMPUTE_BIT,                  &shaders.cs },
      }};

      for (uint32_t i = 0; i < stages.size(); i++) {
        if (stageFlags & stages[i].first) {
          if (!read(*stages[i].second, version))
            return false;
        }
      }

      return true;
    }

    bool read(DxvkBindingMaskV10& data, uint32_t version) {
      // v11 removes this field
      if (version >= 11)
        return true;

      if (version < 9) {
        DxvkBindingMaskV8 v8;
        return read(v8);
      }

      return read(data);
    }

    bool read(DxvkRsInfo& data, uint32_t version) {
      if (version < 13) {
        DxvkRsInfoV12 v12;

        if (!read(v12))
          return false;

        data = v12.convert();
        return true;
      }

      if (version < 14) {
        DxvkRsInfoV13 v13;

        if (!read(v13))
          return false;

        data = v13.convert();
        return true;
      }

      return read(data);
    }

    bool read(DxvkRtInfo& data, uint32_t version) {
      // v12 introduced this field
      if (version < 12)
        return true;

      return read(data);
    }

    bool read(DxvkIlBinding& data, uint32_t version) {
      if (version < 10) {
        DxvkIlBindingV9 v9;

        if (!read(v9))
          return false;

        data = v9.convert();
        return true;
      }

      if (!read(data))
        return false;

      // Format hasn't changed, but we introduced
      // dynamic vertex strides in the meantime
      if (version < 15)
        data.setStride(0);

      return true;
    }


    bool read(DxvkRenderPassFormatV11& data, uint32_t version) {
      uint8_t sampleCount = 0;
      uint8_t imageFormat = 0;
      uint8_t imageLayout = 0;

      if (!read(sampleCount)
       || !read(imageFormat)
       || !read(imageLayout))
        return false;

      data.sampleCount = VkSampleCountFlagBits(sampleCount);
      data.depth.format = VkFormat(imageFormat);
      data.depth.layout = unpackImageLayoutV11(imageLayout);

      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        if (!read(imageFormat)
         || !read(imageLayout))
          return false;

        data.color[i].format = VkFormat(imageFormat);
        data.color[i].layout = unpackImageLayoutV11(imageLayout);
      }

      return true;
    }


    template<typename T>
    bool write(const T& data) {
      if (m_size + sizeof(T) > MaxSize)
        return false;
      
      std::memcpy(&m_data[m_size], &data, sizeof(T));
      m_size += sizeof(T);
      return true;
    }

    bool readFromStream(std::istream& stream, size_t size) {
      if (size > MaxSize)
        return false;

      if (!stream.read(m_data, size))
        return false;

      m_size = size;
      m_read = 0;
      return true;
    }

  private:

    size_t m_size = 0;
    size_t m_read = 0;
    char   m_data[MaxSize];

    template<typename T>
    bool read(T& data) {
      if (m_read + sizeof(T) > m_size)
        return false;

      std::memcpy(&data, &m_data[m_read], sizeof(T));
      m_read += sizeof(T);
      return true;
    }

    static VkImageLayout unpackImageLayoutV11(
            uint8_t                   layout) {
      switch (layout) {
        case 0x80: return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL;
        case 0x81: return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;
        default: return VkImageLayout(layout);
      }
    }

  };


  template<typename T>
  bool readCacheEntryTyped(std::istream& stream, T& entry) {
    auto data = reinterpret_cast<char*>(&entry);
    auto size = sizeof(entry);

    if (!stream.read(data, size))
      return false;
    
    Sha1Hash expectedHash = std::exchange(entry.hash, g_nullHash);
    Sha1Hash computedHash = Sha1Hash::compute(entry);
    return expectedHash == computedHash;
  }


  bool DxvkStateCacheKey::eq(const DxvkStateCacheKey& key) const {
    return this->vs.eq(key.vs)
        && this->tcs.eq(key.tcs)
        && this->tes.eq(key.tes)
        && this->gs.eq(key.gs)
        && this->fs.eq(key.fs)
        && this->cs.eq(key.cs);
  }


  size_t DxvkStateCacheKey::hash() const {
    DxvkHashState hash;
    hash.add(this->vs.hash());
    hash.add(this->tcs.hash());
    hash.add(this->tes.hash());
    hash.add(this->gs.hash());
    hash.add(this->fs.hash());
    hash.add(this->cs.hash());
    return hash;
  }


  DxvkStateCache::DxvkStateCache(
          DxvkDevice*           device,
          DxvkPipelineManager*  pipeManager,
          DxvkPipelineWorkers*  pipeWorkers)
  : m_device      (device),
    m_pipeManager (pipeManager),
    m_pipeWorkers (pipeWorkers) {
    std::string useStateCache = env::getEnvVar("DXVK_STATE_CACHE");
    m_enable = useStateCache != "0" && useStateCache != "disable" &&
      device->config().enableStateCache;

    if (!m_enable)
      return;

    bool newFile = (useStateCache == "reset") || (!readCacheFile());

    if (newFile) {
      auto file = openCacheFileForWrite(true);

      // Write all valid entries to the cache file in
      // case we're recovering a corrupted cache file
      for (auto& e : m_entries)
        writeCacheEntry(file, e);
    }
  }
  

  DxvkStateCache::~DxvkStateCache() {
    this->stopWorkers();
  }


  void DxvkStateCache::addPipelineLibrary(
    const DxvkStateCacheKey&              shaders) {
    if (!m_enable)
      return;

    bool hasAnyStage = false;

    hasAnyStage |= !shaders.vs .eq(g_nullShaderKey);
    hasAnyStage |= !shaders.tcs.eq(g_nullShaderKey);
    hasAnyStage |= !shaders.tes.eq(g_nullShaderKey);
    hasAnyStage |= !shaders.gs .eq(g_nullShaderKey);
    hasAnyStage |= !shaders.fs .eq(g_nullShaderKey);

    if (!hasAnyStage)
      return;

    // Do not add an entry that is already in the cache
    auto entries = m_entryMap.equal_range(shaders);

    for (auto e = entries.first; e != entries.second; e++) {
      if (m_entries[e->second].type == DxvkStateCacheEntryType::PipelineLibrary)
        return;
    }

    // Queue a job to write this pipeline to the cache
    std::unique_lock<dxvk::mutex> lock(m_writerLock);

    m_writerQueue.push({
      DxvkStateCacheEntryType::PipelineLibrary,
      shaders,
      DxvkGraphicsPipelineStateInfo(),
      DxvkComputePipelineStateInfo(),
      0u,
      g_nullHash });
    m_writerCond.notify_one();

    createWriter();
  }


  void DxvkStateCache::addGraphicsPipeline(
    const DxvkStateCacheKey&              shaders,
    const DxvkGraphicsPipelineStateInfo&  state) {
    if (!m_enable || shaders.vs.eq(g_nullShaderKey))
      return;

    // Do not add an entry that is already in the cache
    auto entries = m_entryMap.equal_range(shaders);

    for (auto e = entries.first; e != entries.second; e++) {
      if (m_entries[e->second].type == DxvkStateCacheEntryType::MonolithicPipeline
       && m_entries[e->second].gpState == state)
        return;
    }

    // Queue a job to write this pipeline to the cache
    std::unique_lock<dxvk::mutex> lock(m_writerLock);

    m_writerQueue.push({
      DxvkStateCacheEntryType::MonolithicPipeline,
      shaders,
      state,
      DxvkComputePipelineStateInfo(),
      0u,
      g_nullHash });
    m_writerCond.notify_one();

    createWriter();
  }


  void DxvkStateCache::addComputePipeline(
    const DxvkShaderKey&                  shader,
          uint32_t                        specConstantMask,
    const DxvkComputePipelineStateInfo&   state) {
    if (!m_enable)
      return;

    if (shader.eq(g_nullShaderKey))
      return;

    specConstantMask &= (1u << MaxNumSpecConstants) - 1u;

    DxvkStateCacheKey shaders;
    shaders.cs = shader;

    auto entries = m_entryMap.equal_range(shaders);

    for (auto e = entries.first; e != entries.second; e++) {
      const auto& existing = m_entries[e->second];

      if (existing.type == DxvkStateCacheEntryType::ComputePipeline
       && existing.cpSpecConstantMask == specConstantMask
       && existing.cpState == state) {
        Logger::ehang(str::format(
          "Skipping duplicate compute pipeline variant for ",
          describeComputeState(shaders, specConstantMask, state)));
        return;
      }
    }

    std::unique_lock<dxvk::mutex> lock(m_writerLock);

    m_writerQueue.push({
      DxvkStateCacheEntryType::ComputePipeline,
      shaders,
      DxvkGraphicsPipelineStateInfo(),
      state,
      specConstantMask,
      g_nullHash });

    Logger::ehang(str::format(
      "Queued compute pipeline state cache entry for ",
      describeComputeState(shaders, specConstantMask, state)));

    m_writerCond.notify_one();

    createWriter();
  }


  void DxvkStateCache::registerShader(const Rc<DxvkShader>& shader) {
    if (!m_enable)
      return;

    DxvkShaderKey key = shader->getShaderKey();

    if (key.eq(g_nullShaderKey))
      return;
    
    // Add the shader so we can look it up by its key
    std::unique_lock<dxvk::mutex> entryLock(m_entryLock);
    m_shaderMap.insert({ key, shader });

    // Deferred lock, don't stall workers unless we have to
    std::unique_lock<dxvk::mutex> workerLock;

    auto pipelines = m_pipelineMap.equal_range(key);

    for (auto p = pipelines.first; p != pipelines.second; p++) {
      bool hasGraphicsStage = !p->second.vs .eq(g_nullShaderKey)
                           || !p->second.tcs.eq(g_nullShaderKey)
                           || !p->second.tes.eq(g_nullShaderKey)
                           || !p->second.gs .eq(g_nullShaderKey)
                           || !p->second.fs .eq(g_nullShaderKey);

      bool hasComputeStage = !p->second.cs.eq(g_nullShaderKey);

      if (hasComputeStage && !hasGraphicsStage) {
        Rc<DxvkShader> csShader;

        if (!getShaderByKey(p->second.cs, csShader))
          continue;

        compileComputePipelines(csShader, p->second);
        continue;
      }

      WorkerItem item;

      if (!getShaderByKey(p->second.vs,  item.gp.vs)
       || !getShaderByKey(p->second.tcs, item.gp.tcs)
       || !getShaderByKey(p->second.tes, item.gp.tes)
       || !getShaderByKey(p->second.gs,  item.gp.gs)
       || !getShaderByKey(p->second.fs,  item.gp.fs))
        continue;
      
      if (!workerLock)
        workerLock = std::unique_lock<dxvk::mutex>(m_workerLock);
      
      m_workerQueue.push(item);
    }

    if (workerLock) {
      m_workerCond.notify_all();
      createWorker();
    }
  }


  void DxvkStateCache::registerCachedShader(const Rc<DxvkShader>& shader) {
    if (!m_enable)
      return;

    DxvkShaderKey key = shader->getShaderKey();

    if (key.eq(g_nullShaderKey))
      return;

    std::lock_guard<dxvk::mutex> entryLock(m_entryLock);
    m_shaderMap.insert({ key, shader });
  }


  void DxvkStateCache::prewarmAllPipelines() {
    if (!m_enable)
      return;

    auto formatShaderSet = [] (const DxvkStateCacheKey& key) {
      std::string result;

      auto append = [&result] (const char* name, const DxvkShaderKey& shader) {
        if (shader.eq(g_nullShaderKey))
          return;

        if (!result.empty())
          result += ", ";

        result += name;
        result += '=';
        result += shader.toString();
      };

      append("VS",  key.vs);
      append("TCS", key.tcs);
      append("TES", key.tes);
      append("GS",  key.gs);
      append("FS",  key.fs);
      append("CS",  key.cs);

      if (result.empty())
        result = "<no stages>";

      return result;
    };

    std::unordered_map<DxvkStateCacheKey, WorkerItem, DxvkHash, DxvkEq> workItems;
    workItems.reserve(m_entries.size());

    std::unordered_map<DxvkStateCacheKey, Rc<DxvkShader>, DxvkHash, DxvkEq> computeWorkItems;
    computeWorkItems.reserve(m_entries.size());

    size_t skippedGraphicsSets = 0;
    size_t skippedComputeSets = 0;

    for (const auto& entry : m_entries) {
      const DxvkStateCacheKey& key = entry.shaders;

      if (workItems.find(key) != workItems.end()
       || computeWorkItems.find(key) != computeWorkItems.end())
        continue;

      bool hasGraphicsStage = !key.vs .eq(g_nullShaderKey)
                           || !key.tcs.eq(g_nullShaderKey)
                           || !key.tes.eq(g_nullShaderKey)
                           || !key.gs .eq(g_nullShaderKey)
                           || !key.fs .eq(g_nullShaderKey);

      bool hasComputeStage = !key.cs.eq(g_nullShaderKey);

      if (entry.type == DxvkStateCacheEntryType::ComputePipeline
       || (hasComputeStage && !hasGraphicsStage)) {
        Rc<DxvkShader> csShader;

        if (!getShaderByKey(key.cs, csShader)) {
          Logger::ehang(str::format(
            "Skipping cached compute shader set ", formatShaderSet(key),
            " because compute stage with key ", key.cs.toString(),
            " is unavailable"));
          skippedComputeSets += 1;
          continue;
        }

        computeWorkItems.emplace(key, std::move(csShader));
        continue;
      }

      WorkerItem item;
      bool missingStage = false;

      auto ensureShader = [&] (const DxvkShaderKey& shaderKey,
        Rc<DxvkShader>& shaderObject, const char* stageName) {
        if (shaderKey.eq(g_nullShaderKey))
          return;

        if (!getShaderByKey(shaderKey, shaderObject)) {
          Logger::ehang(str::format(
            "Skipping cached shader set ", formatShaderSet(key),
            " because stage ", stageName,
            " with key ", shaderKey.toString(),
            " is unavailable"));
          missingStage = true;
        }
      };

      ensureShader(key.vs,  item.gp.vs,  "VS");
      ensureShader(key.tcs, item.gp.tcs, "TCS");
      ensureShader(key.tes, item.gp.tes, "TES");
      ensureShader(key.gs,  item.gp.gs,  "GS");
      ensureShader(key.fs,  item.gp.fs,  "FS");

      if (missingStage) {
        skippedGraphicsSets += 1;
        continue;
      }

      workItems.emplace(key, std::move(item));
    }

    if (workItems.empty() && computeWorkItems.empty()) {
      Logger::ehang("No cached pipelines available in state cache for prewarm");
      return;
    }

    size_t pipelineLibraryCount = 0;
    size_t monolithicPipelineCount = 0;
    size_t computePipelineCount = 0;

    for (const auto& entry : workItems) {
      auto range = m_entryMap.equal_range(entry.first);

      for (auto it = range.first; it != range.second; ++it) {
        const auto& cacheEntry = m_entries[it->second];

        if (cacheEntry.type == DxvkStateCacheEntryType::PipelineLibrary)
          pipelineLibraryCount += 1;
        else if (cacheEntry.type == DxvkStateCacheEntryType::MonolithicPipeline)
          monolithicPipelineCount += 1;
      }
    }

    for (const auto& entry : computeWorkItems) {
      auto range = m_entryMap.equal_range(entry.first);

      for (auto it = range.first; it != range.second; ++it) {
        const auto& cacheEntry = m_entries[it->second];

        if (cacheEntry.type == DxvkStateCacheEntryType::ComputePipeline)
          computePipelineCount += 1;
      }
    }

    Logger::ehang(str::format(
      "Prewarming ", workItems.size(),
      " graphics shader sets and ", computeWorkItems.size(),
      " compute shader sets from state cache (",
      pipelineLibraryCount, " pipeline libraries, ",
      monolithicPipelineCount, " monolithic pipelines, ",
      computePipelineCount, " compute pipelines)"));

    if (skippedGraphicsSets)
      Logger::ehang(str::format(
        "Skipped ", skippedGraphicsSets,
        " shader sets because required stages were missing from cache"));

    if (skippedComputeSets)
      Logger::ehang(str::format(
        "Skipped ", skippedComputeSets,
        " compute shader sets because required stages were missing from cache"));

    for (const auto& item : workItems)
      compilePipelines(item.second);

    for (const auto& item : computeWorkItems)
      compileComputePipelines(item.second, item.first);
  }


  void DxvkStateCache::stopWorkers() {
    { std::lock_guard<dxvk::mutex> workerLock(m_workerLock);
      std::lock_guard<dxvk::mutex> writerLock(m_writerLock);

      if (m_stopThreads.exchange(true))
        return;

      m_workerCond.notify_all();
      m_writerCond.notify_all();
    }

    if (m_workerThread.joinable())
      m_workerThread.join();
    
    if (m_writerThread.joinable())
      m_writerThread.join();
  }


  DxvkShaderKey DxvkStateCache::getShaderKey(const Rc<DxvkShader>& shader) const {
    return shader != nullptr ? shader->getShaderKey() : g_nullShaderKey;
  }


  bool DxvkStateCache::getShaderByKey(
    const DxvkShaderKey&            key,
          Rc<DxvkShader>&           shader) const {
    if (key.eq(g_nullShaderKey))
      return true;
    
    auto entry = m_shaderMap.find(key);
    if (entry == m_shaderMap.end())
      return false;

    shader = entry->second;
    return true;
  }


  void DxvkStateCache::mapPipelineToEntry(
    const DxvkStateCacheKey&        key,
          size_t                    entryId) {
    m_entryMap.insert({ key, entryId });
  }

  
  void DxvkStateCache::mapShaderToPipeline(
    const DxvkShaderKey&            shader,
    const DxvkStateCacheKey&        key) {
    if (!shader.eq(g_nullShaderKey))
      m_pipelineMap.insert({ shader, key });
  }


  void DxvkStateCache::compilePipelines(const WorkerItem& item) {
    DxvkStateCacheKey key = { };
    key.vs  = getShaderKey(item.gp.vs);
    key.tcs = getShaderKey(item.gp.tcs);
    key.tes = getShaderKey(item.gp.tes);
    key.gs  = getShaderKey(item.gp.gs);
    key.fs  = getShaderKey(item.gp.fs);
    key.cs  = g_nullShaderKey;

    DxvkGraphicsPipeline* pipeline = nullptr;
    auto entries = m_entryMap.equal_range(key);

    for (auto e = entries.first; e != entries.second; e++) {
      const auto& entry = m_entries[e->second];

      auto describeShaders = [&entry] () {
        std::string description;

        auto append = [&] (const char* name, const DxvkShaderKey& shader) {
          if (shader.eq(g_nullShaderKey))
            return;

          if (!description.empty())
            description += ", ";

          description += name;
          description += '=';
          description += shader.toString();
        };

        append("VS",  entry.shaders.vs);
        append("TCS", entry.shaders.tcs);
        append("TES", entry.shaders.tes);
        append("GS",  entry.shaders.gs);
        append("FS",  entry.shaders.fs);
        append("CS",  entry.shaders.cs);

        if (description.empty())
          description = "<no stages>";

        return description;
      };

      switch (entry.type) {
        case DxvkStateCacheEntryType::MonolithicPipeline: {
          if (!pipeline)
            pipeline = m_pipeManager->createGraphicsPipeline(item.gp);

          if (!pipeline) {
            Logger::ehang(str::format(
              "Failed to acquire graphics pipeline for cached state ",
              describeShaders()));
            continue;
          }

          if (m_device->canUseGraphicsPipelineLibrary()) {
            if (item.gp.vs != nullptr) {
              DxvkGraphicsPipelineVertexInputState viState(m_device, entry.gpState, item.gp.vs.ptr());
              m_pipeManager->createVertexInputLibrary(viState);
            }

            if (item.gp.fs != nullptr) {
              DxvkGraphicsPipelineFragmentOutputState foState(m_device, entry.gpState, item.gp.fs.ptr());
              m_pipeManager->createFragmentOutputLibrary(foState);
            }

            auto handleInfo = pipeline->getPipelineHandle(entry.gpState);

            if (!handleInfo.first) {
              Logger::ehang(str::format(
                "Failed to prewarm graphics pipeline for shader set ",
                describeShaders()));
            } else {
              Logger::ehang(str::format(
                "Prewarmed cached graphics pipeline for shader set ",
                describeShaders(),
                " via getPipelineHandle"));
            }
          } else {
            pipeline->compilePipeline(entry.gpState);
            Logger::ehang(str::format(
              "Prewarmed cached graphics pipeline for shader set ",
              describeShaders(),
              " via compilePipeline"));
          }
        } break;

        case DxvkStateCacheEntryType::PipelineLibrary: {
          if (!m_device->canUseGraphicsPipelineLibrary())
            break;

          DxvkShaderPipelineLibraryKey libraryKey;

          auto tryAddShader = [&] (const DxvkShaderKey& shaderKey, const Rc<DxvkShader>& shaderObject) {
            if (!shaderKey.eq(g_nullShaderKey) && shaderObject != nullptr)
              libraryKey.addShader(shaderObject);
          };

          tryAddShader(entry.shaders.vs,  item.gp.vs);
          tryAddShader(entry.shaders.tcs, item.gp.tcs);
          tryAddShader(entry.shaders.tes, item.gp.tes);
          tryAddShader(entry.shaders.gs,  item.gp.gs);
          tryAddShader(entry.shaders.fs,  item.gp.fs);

          if (!libraryKey.canUsePipelineLibrary())
            break;

          auto pipelineLibrary = m_pipeManager->createShaderPipelineLibrary(libraryKey);

          if (pipelineLibrary) {
            pipelineLibrary->compilePipeline();
            Logger::ehang(str::format(
              "Prewarmed cached shader pipeline library for shader set ",
              describeShaders()));
          }
        } break;
      }
    }
  }


  void DxvkStateCache::compileComputePipelines(
    const Rc<DxvkShader>&           shader,
    const DxvkStateCacheKey&        key) {
    DxvkComputePipelineShaders shaders;
    shaders.cs = shader;

    auto entries = m_entryMap.equal_range(key);

    for (auto e = entries.first; e != entries.second; ++e) {
      const auto& entry = m_entries[e->second];

      if (entry.type != DxvkStateCacheEntryType::ComputePipeline)
        continue;

      auto pipeline = m_pipeManager->createComputePipeline(shaders);

      if (!pipeline) {
        Logger::ehang(str::format(
          "Failed to acquire compute pipeline for cached shader ",
          key.cs.toString()));
        continue;
      }

      Logger::ehang(str::format(
        "Prewarming cached compute pipeline variant for ",
        describeComputeState(key, entry.cpSpecConstantMask, entry.cpState)));

      VkPipeline handle = pipeline->getPipelineHandle(entry.cpState);

      if (handle) {
        Logger::ehang(str::format(
          "Prewarmed cached compute pipeline for ",
          describeComputeState(key, entry.cpSpecConstantMask, entry.cpState)));
      } else {
        Logger::ehang(str::format(
          "Failed to prewarm compute pipeline for ",
          describeComputeState(key, entry.cpSpecConstantMask, entry.cpState)));
      }
    }
  }


  bool DxvkStateCache::readCacheFile() {
    // Return success if the file was not found.
    // This way we will only create it on demand.
    std::ifstream ifile = openCacheFileForRead();

    if (!ifile) {
      Logger::warn("DXVK: No state cache file found");
      return true;
    }

    // The header stores the state cache version,
    // we need to regenerate it if it's outdated
    DxvkStateCacheHeader newHeader;
    DxvkStateCacheHeader curHeader;

    if (!readCacheHeader(ifile, curHeader)) {
      Logger::warn("DXVK: Failed to read state cache header");
      return false;
    }

    // Discard caches of unsupported versions
    if (curHeader.version < 8 || curHeader.version == 16
     || curHeader.version > newHeader.version) {
      Logger::warn("DXVK: State cache version not supported");
      return false;
    }

    // Notify user about format conversion
    if (curHeader.version != newHeader.version)
      Logger::warn(str::format("DXVK: Updating state cache version to v", newHeader.version));

    // Read actual cache entries from the file.
    // If we encounter invalid entries, we should
    // regenerate the entire state cache file.
    uint32_t numInvalidEntries = 0;

    while (ifile) {
      DxvkStateCacheEntry entry;

      if (readCacheEntry(curHeader.version, ifile, entry)) {
        size_t entryId = m_entries.size();
        m_entries.push_back(entry);

        mapPipelineToEntry(entry.shaders, entryId);

        mapShaderToPipeline(entry.shaders.vs,  entry.shaders);
        mapShaderToPipeline(entry.shaders.tcs, entry.shaders);
        mapShaderToPipeline(entry.shaders.tes, entry.shaders);
        mapShaderToPipeline(entry.shaders.gs,  entry.shaders);
        mapShaderToPipeline(entry.shaders.fs,  entry.shaders);
        mapShaderToPipeline(entry.shaders.cs,  entry.shaders);
      } else if (ifile) {
        numInvalidEntries += 1;
      }
    }

    Logger::info(str::format(
      "DXVK: Read ", m_entries.size(),
      " valid state cache entries"));

    if (numInvalidEntries) {
      Logger::warn(str::format(
        "DXVK: Skipped ", numInvalidEntries,
        " invalid state cache entries"));
      return false;
    }
    
    // Rewrite entire state cache if it is outdated
    return curHeader.version == newHeader.version;
  }


  bool DxvkStateCache::readCacheHeader(
          std::istream&             stream,
          DxvkStateCacheHeader&     header) const {
    DxvkStateCacheHeader expected;

    auto data = reinterpret_cast<char*>(&header);
    auto size = sizeof(header);

    if (!stream.read(data, size))
      return false;
    
    for (uint32_t i = 0; i < 4; i++) {
      if (expected.magic[i] != header.magic[i])
        return false;
    }
    
    return true;
  }


  bool DxvkStateCache::readCacheEntry(
          uint32_t                  version,
          std::istream&             stream,
          DxvkStateCacheEntry&      entry) const {
    // Read entry metadata and actual data
    DxvkStateCacheEntryData data;
    VkShaderStageFlags stageMask;
    Sha1Hash hash;
    uint32_t entryTypeValue = uint32_t(DxvkStateCacheEntryType::MonolithicPipeline);
    uint32_t entrySize = 0;

    if (version >= 18) {
      DxvkStateCacheEntryHeader header;

      if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header)))
        return false;

      entryTypeValue = header.entryType;
      stageMask = VkShaderStageFlags(header.stageMask);
      entrySize = header.entrySize;
    } else if (version >= 16) {
      DxvkStateCacheEntryHeaderV16 header;

      if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header)))
        return false;

      entryTypeValue = header.entryType;
      stageMask = VkShaderStageFlags(header.stageMask);
      entrySize = header.entrySize;
    } else {
      DxvkStateCacheEntryHeaderV8 headerV8;

      if (!stream.read(reinterpret_cast<char*>(&headerV8), sizeof(headerV8)))
        return false;

      entryTypeValue = uint32_t(DxvkStateCacheEntryType::MonolithicPipeline);
      stageMask = VkShaderStageFlags(headerV8.stageMask);
      entrySize = headerV8.entrySize;
    }

    if (!stream.read(reinterpret_cast<char*>(&hash), sizeof(hash))
     || !data.readFromStream(stream, entrySize))
      return false;

    // Validate hash, skip entry if invalid
    if (hash != data.computeHash())
      return false;

    // Set up entry metadata
    entry.type = DxvkStateCacheEntryType(entryTypeValue);
    entry.gpState = DxvkGraphicsPipelineStateInfo();
    entry.cpState = DxvkComputePipelineStateInfo();
    entry.cpSpecConstantMask = 0;

    if (entry.type == DxvkStateCacheEntryType::ComputePipeline && version < 19)
      return false;

    // Read shader hashes
    auto entryType = DxvkStateCacheEntryType(entryTypeValue);
    data.read(entry.shaders, version, stageMask);

    if (entryType == DxvkStateCacheEntryType::PipelineLibrary)
      return true;

    DxvkBindingMaskV10 dummyBindingMask = { };
    bool isCompute = stageMask & VK_SHADER_STAGE_COMPUTE_BIT;

    if (isCompute) {
      if (!data.read(dummyBindingMask, version))
        return false;

      if (version >= 19) {
        if (!data.read(entry.cpSpecConstantMask, version))
          return false;

        entry.cpSpecConstantMask &= (1u << MaxNumSpecConstants) - 1u;
      }
    } else {
      // Read packed render pass format
      if (version < 12) {
        DxvkRenderPassFormatV11 v11;
        data.read(v11, version);
        entry.gpState.rt = v11.convert();
      }

      // Read common pipeline state
      if (!data.read(dummyBindingMask, version)
       || !data.read(entry.gpState.ia, version)
       || !data.read(entry.gpState.il, version)
       || !data.read(entry.gpState.rs, version)
       || !data.read(entry.gpState.ms, version)
       || !data.read(entry.gpState.ds, version)
       || !data.read(entry.gpState.om, version)
       || !data.read(entry.gpState.rt, version)
       || !data.read(entry.gpState.dsFront, version)
       || !data.read(entry.gpState.dsBack, version))
        return false;

      if (entry.gpState.il.attributeCount() > MaxNumVertexAttributes
       || entry.gpState.il.bindingCount() > MaxNumVertexBindings)
        return false;

      // Read render target swizzles
      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        if (!data.read(entry.gpState.omSwizzle[i], version))
          return false;
      }

      // Read render target blend info
      for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
        if (!data.read(entry.gpState.omBlend[i], version))
          return false;
      }

      // Read defined vertex attributes
      for (uint32_t i = 0; i < entry.gpState.il.attributeCount(); i++) {
        if (!data.read(entry.gpState.ilAttributes[i], version))
          return false;
      }

      // Read defined vertex bindings
      for (uint32_t i = 0; i < entry.gpState.il.bindingCount(); i++) {
        if (!data.read(entry.gpState.ilBindings[i], version))
          return false;
      }
    }

    // Read non-zero spec constants
    uint32_t specConstantMask = 0;

    if (isCompute) {
      specConstantMask = entry.cpSpecConstantMask;
    } else {
      if (!data.read(specConstantMask, version))
        return false;
    }

    for (uint32_t i = 0; i < MaxNumSpecConstants; i++) {
      if (specConstantMask & (1 << i)) {
        uint32_t value = 0;

        if (!data.read(value, version))
          return false;

        if (isCompute)
          entry.cpState.sc.specConstants[i] = value;
        else
          entry.gpState.sc.specConstants[i] = value;
      }
    }

    return true;
  }


  void DxvkStateCache::writeCacheEntry(
          std::ostream&             stream,
          DxvkStateCacheEntry&      entry) const {
    DxvkStateCacheEntryData data;
    VkShaderStageFlags stageMask = 0;

    // Write shader hashes
    std::array<std::pair<VkShaderStageFlagBits, const DxvkShaderKey*>, 6> stages = {{
      { VK_SHADER_STAGE_VERTEX_BIT,                   &entry.shaders.vs },
      { VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,     &entry.shaders.tcs },
      { VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,  &entry.shaders.tes },
      { VK_SHADER_STAGE_GEOMETRY_BIT,                 &entry.shaders.gs },
      { VK_SHADER_STAGE_FRAGMENT_BIT,                 &entry.shaders.fs },
      { VK_SHADER_STAGE_COMPUTE_BIT,                  &entry.shaders.cs },
    }};

    for (uint32_t i = 0; i < stages.size(); i++) {
      if (!stages[i].second->eq(g_nullShaderKey)) {
        stageMask |= stages[i].first;
        data.write(*stages[i].second);
      }
    }

    if (entry.type == DxvkStateCacheEntryType::PipelineLibrary) {
      // Nothing else to write
    } else if (entry.type == DxvkStateCacheEntryType::ComputePipeline) {
      DxvkBindingMaskV10 dummyBindingMask = { };
      data.write(dummyBindingMask);

      uint32_t specConstantMask = entry.cpSpecConstantMask & ((1u << MaxNumSpecConstants) - 1u);

      data.write(specConstantMask);

      for (uint32_t i = 0; i < MaxNumSpecConstants; i++) {
        if (specConstantMask & (1 << i))
          data.write(entry.cpState.sc.specConstants[i]);
      }
    } else {
      // Write out common pipeline state
      data.write(entry.gpState.ia);
      data.write(entry.gpState.il);
      data.write(entry.gpState.rs);
      data.write(entry.gpState.ms);
      data.write(entry.gpState.ds);
      data.write(entry.gpState.om);
      data.write(entry.gpState.rt);
      data.write(entry.gpState.dsFront);
      data.write(entry.gpState.dsBack);

      // Write out render target swizzles and blend info
      for (uint32_t i = 0; i < MaxNumRenderTargets; i++)
        data.write(entry.gpState.omSwizzle[i]);

      for (uint32_t i = 0; i < MaxNumRenderTargets; i++)
        data.write(entry.gpState.omBlend[i]);

      // Write out input layout for defined attributes
      for (uint32_t i = 0; i < entry.gpState.il.attributeCount(); i++)
        data.write(entry.gpState.ilAttributes[i]);

      for (uint32_t i = 0; i < entry.gpState.il.bindingCount(); i++)
        data.write(entry.gpState.ilBindings[i]);

      // Write out all non-zero spec constants
      uint32_t specConstantMask = 0;

      for (uint32_t i = 0; i < MaxNumSpecConstants; i++)
        specConstantMask |= entry.gpState.sc.specConstants[i] ? (1 << i) : 0;

      data.write(specConstantMask);

      for (uint32_t i = 0; i < MaxNumSpecConstants; i++) {
        if (specConstantMask & (1 << i))
          data.write(entry.gpState.sc.specConstants[i]);
      }
    }

    // General layout: header -> hash -> data
    DxvkStateCacheEntryHeader header;
    header.entryType = uint32_t(entry.type);
    header.stageMask = uint32_t(stageMask);
    header.entrySize = data.size();

    Sha1Hash hash = data.computeHash();

    stream.write(reinterpret_cast<char*>(&header), sizeof(header));
    stream.write(reinterpret_cast<char*>(&hash), sizeof(hash));
    stream.write(data.data(), data.size());
    stream.flush();
  }


  void DxvkStateCache::workerFunc() {
    env::setThreadName("dxvk-worker");

    while (!m_stopThreads.load()) {
      WorkerItem item;

      { std::unique_lock<dxvk::mutex> lock(m_workerLock);

        if (m_workerQueue.empty()) {
          m_workerCond.wait(lock, [this] () {
            return m_workerQueue.size()
                || m_stopThreads.load();
          });
        }

        if (m_workerQueue.empty())
          break;
        
        item = m_workerQueue.front();
        m_workerQueue.pop();
      }

      compilePipelines(item);
    }
  }


  void DxvkStateCache::writerFunc() {
    env::setThreadName("dxvk-writer");

    std::ofstream file;

    while (!m_stopThreads.load()) {
      DxvkStateCacheEntry entry;

      { std::unique_lock<dxvk::mutex> lock(m_writerLock);

        m_writerCond.wait(lock, [this] () {
          return m_writerQueue.size()
              || m_stopThreads.load();
        });

        if (m_writerQueue.size() == 0)
          break;

        entry = m_writerQueue.front();
        m_writerQueue.pop();
      }

      if (!file.is_open())
        file = openCacheFileForWrite(false);

      writeCacheEntry(file, entry);
    }
  }


  void DxvkStateCache::createWorker() {
    if (!m_workerThread.joinable())
      m_workerThread = dxvk::thread([this] () { workerFunc(); });
  }


  void DxvkStateCache::createWriter() {
    if (!m_writerThread.joinable())
      m_writerThread = dxvk::thread([this] () { writerFunc(); });
  }


  str::path_string DxvkStateCache::getCacheFileName() const {
    std::string path = getCacheDir();

    if (!path.empty() && *path.rbegin() != '/')
      path += '/';
    
    std::string exeName = env::getExeBaseName();
    path += exeName + ".dxvk-cache";
    return str::topath(path.c_str());
  }


  std::ifstream DxvkStateCache::openCacheFileForRead() const {
    return std::ifstream(getCacheFileName().c_str(), std::ios_base::binary);
  }


  std::ofstream DxvkStateCache::openCacheFileForWrite(bool recreate) const {
    std::ofstream file;

    if (!recreate) {
      // Apparently there's no other way to check whether
      // the file is empty after creating an ofstream
      recreate = !openCacheFileForRead();
    }

    if (recreate) {
      file = std::ofstream(getCacheFileName().c_str(),
        std::ios_base::binary |
        std::ios_base::trunc);

      if (!file && env::createDirectory(getCacheDir())) {
        file = std::ofstream(getCacheFileName().c_str(),
          std::ios_base::binary |
          std::ios_base::trunc);
      }
    } else {
      file = std::ofstream(getCacheFileName().c_str(),
        std::ios_base::binary |
        std::ios_base::app);
    }

    if (!file)
      return file;

    if (recreate) {
      Logger::warn("DXVK: Creating new state cache file");

      // Write header with the current version number
      DxvkStateCacheHeader header;

      auto data = reinterpret_cast<const char*>(&header);
      auto size = sizeof(header);

      file.write(data, size);
    }

    return file;
  }


  std::string DxvkStateCache::getCacheDir() const {
    return env::getEnvVar("DXVK_STATE_CACHE_PATH");
  }

}
