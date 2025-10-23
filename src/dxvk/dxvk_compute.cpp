#include <cstring>
#include <sstream>

#include "../util/util_time.h"

#include "dxvk_compute.h"
#include "dxvk_device.h"
#include "dxvk_log_util.h"
#include "dxvk_pipemanager.h"
#include "dxvk_spec_const.h"
#include "dxvk_state_cache.h"

namespace dxvk {
  
  DxvkComputePipeline::DxvkComputePipeline(
          DxvkPipelineManager*        pipeMgr,
          DxvkComputePipelineShaders  shaders)
  : m_vkd(pipeMgr->m_device->vkd()), m_pipeMgr(pipeMgr),
    m_shaders(std::move(shaders)) {
    m_shaders.cs->defineResourceSlots(m_slotMapping);

    m_slotMapping.makeDescriptorsDynamic(
      m_pipeMgr->m_device->options().maxNumDynamicUniformBuffers,
      m_pipeMgr->m_device->options().maxNumDynamicStorageBuffers);
    
    m_layout = new DxvkPipelineLayout(m_vkd,
      m_slotMapping, VK_PIPELINE_BIND_POINT_COMPUTE);
  }
  
  
  DxvkComputePipeline::~DxvkComputePipeline() {
    for (const auto& instance : m_pipelines)
      this->destroyPipeline(instance.pipeline());
  }
  
  
  VkPipeline DxvkComputePipeline::getPipelineHandle(
    const DxvkComputePipelineStateInfo& state) {
    DxvkComputePipelineInstance* instance = this->findInstance(state);

    if (unlikely(!instance)) {
      std::lock_guard<dxvk::mutex> lock(m_mutex);
      instance = this->findInstance(state);

      if (!instance) {
        instance = this->createInstance(state);

        if (instance != nullptr)
          this->writePipelineStateToCache(state);
      }
    }

    if (!instance)
      return VK_NULL_HANDLE;

    return instance->pipeline();
  }


  void DxvkComputePipeline::compilePipeline(
    const DxvkComputePipelineStateInfo& state) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    if (!this->findInstance(state))
      this->createInstance(state);
  }
  
  
  DxvkComputePipelineInstance* DxvkComputePipeline::createInstance(
    const DxvkComputePipelineStateInfo& state) {
    const uint64_t pipelineId = m_pipeMgr->allocatePipelineId();
    VkPipeline newPipelineHandle = this->createPipeline(state, pipelineId);

    if (newPipelineHandle == VK_NULL_HANDLE)
      return nullptr;

    m_pipeMgr->m_numComputePipelines += 1;
    return &(*m_pipelines.emplace(state, pipelineId, newPipelineHandle));
  }

  
  DxvkComputePipelineInstance* DxvkComputePipeline::findInstance(
    const DxvkComputePipelineStateInfo& state) {
    for (auto& instance : m_pipelines) {
      if (instance.isCompatible(state))
        return &instance;
    }
    
    return nullptr;
  }
  
  
  VkPipeline DxvkComputePipeline::createPipeline(
    const DxvkComputePipelineStateInfo& state,
          uint64_t                      pipelineId) const {
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    auto stringifyBindingMask = [&state] () {
      std::ostringstream ss;
      ss << "[";
      bool first = true;
      for (int32_t binding = state.bsBindingMask.findNext(0);
           binding >= 0;
           binding = state.bsBindingMask.findNext(binding + 1)) {
        if (!first)
          ss << ",";
        first = false;
        ss << binding;
      }
      ss << "]";
      return ss.str();
    };

    auto stringifySpecConstants = [&state] () {
      std::ostringstream ss;
      ss << "[";
      for (uint32_t i = 0; i < DxvkLimits::MaxNumSpecConstants; i++) {
        if (i)
          ss << ",";
        ss << state.sc.specConstants[i];
      }
      ss << "]";
      return ss.str();
    };

    const std::string shaderName = m_shaders.cs != nullptr
      ? m_shaders.cs->debugName()
      : std::string("<null>");

    Logger::info(log::ehang(
      "Compiling compute pipeline #", pipelineId,
      " cs=", shaderName,
      " bindings=", stringifyBindingMask(),
      " specConstants=", stringifySpecConstants()));

    DxvkSpecConstants specData;
    for (uint32_t i = 0; i < m_layout->bindingCount(); i++)
      specData.set(i, state.bsBindingMask.test(i), true);
    
    for (uint32_t i = 0; i < MaxNumSpecConstants; i++)
      specData.set(getSpecId(i), state.sc.specConstants[i], 0u);

    VkSpecializationInfo specInfo = specData.getSpecInfo();
    
    DxvkShaderModuleCreateInfo moduleInfo;
    moduleInfo.fsDualSrcBlend = false;

    auto csm = m_shaders.cs->createShaderModule(m_vkd, m_slotMapping, moduleInfo);

    VkComputePipelineCreateInfo info;
    info.sType                = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.pNext                = nullptr;
    info.flags                = 0;
    info.stage                = csm.stageInfo(&specInfo);
    info.layout               = m_layout->pipelineLayout();
    info.basePipelineHandle   = VK_NULL_HANDLE;
    info.basePipelineIndex    = -1;
    
    const auto compileStart = dxvk::high_resolution_clock::now();

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (m_vkd->vkCreateComputePipelines(m_vkd->device(),
          m_pipeMgr->m_cache->handle(), 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
      Logger::err("DxvkComputePipeline: Failed to compile pipeline");
      Logger::err(log::ehang("Compute pipeline #", pipelineId,
        " failed cs=", shaderName));
      return VK_NULL_HANDLE;
    }

    const auto compileEnd = dxvk::high_resolution_clock::now();
    const auto compileDuration = std::chrono::duration<double, std::milli>(compileEnd - compileStart);
    Logger::info(log::ehang("Compute pipeline #", pipelineId,
      " compiled in ", compileDuration.count(), " ms"));

    return pipeline;
  }


  void DxvkComputePipeline::destroyPipeline(VkPipeline pipeline) {
    m_vkd->vkDestroyPipeline(m_vkd->device(), pipeline, nullptr);
  }
  
  
  void DxvkComputePipeline::writePipelineStateToCache(
    const DxvkComputePipelineStateInfo& state) const {
    if (m_pipeMgr->m_stateCache == nullptr)
      return;
    
    DxvkStateCacheKey key;

    if (m_shaders.cs != nullptr)
      key.cs = m_shaders.cs->getShaderKey();

    m_pipeMgr->m_stateCache->addComputePipeline(key, state);
  }
  
}
