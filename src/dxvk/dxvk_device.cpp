#include <array>
#include <memory>
#include <vector>

#include <dxvk_fullscreen_vert.h>

#include "../spirv/spirv_instruction.h"

#include "dxvk_device.h"
#include "dxvk_instance.h"
#include "dxvk_pipelayout.h"

#include "../util/log/log.h"
#include "../util/util_error.h"

namespace dxvk {
  
  DxvkDevice::DxvkDevice(
    const Rc<DxvkInstance>&         instance,
    const Rc<DxvkAdapter>&          adapter,
    const Rc<vk::DeviceFn>&         vkd,
    const DxvkDeviceFeatures&       features,
    const DxvkDeviceQueueSet&       queues,
    const DxvkQueueCallback&        queueCallback)
  : m_options           (instance->options()),
    m_instance          (instance),
    m_adapter           (adapter),
    m_vkd               (vkd),
    m_features          (features),
    m_properties        (adapter->devicePropertiesExt()),
    m_perfHints         (getPerfHints()),
    m_objects           (this),
    m_queues            (queues),
    m_submissionQueue   (this, queueCallback) {

    m_shaderCache = std::make_unique<DxvkShaderCache>(this);
    m_shaderCache->initialize();

  }
  
  
  DxvkDevice::~DxvkDevice() {
    // If we are being destroyed during/after DLL process detachment
    // from TerminateProcess, etc, our CS threads are already destroyed
    // and we cannot synchronize against them.
    // The best we can do is just wait for the Vulkan device to be idle.
    if (this_thread::isInModuleDetachment())
      return;

    // Wait for all pending Vulkan commands to be
    // executed before we destroy any resources.
    this->waitForIdle();

    // Stop workers explicitly in order to prevent
    // access to structures that are being destroyed.
    m_objects.pipelineManager().stopWorkerThreads();
  }


  bool DxvkDevice::isUnifiedMemoryArchitecture() const {
    return m_adapter->isUnifiedMemoryArchitecture();
  }


  bool DxvkDevice::canUseGraphicsPipelineLibrary() const {
    // Without graphicsPipelineLibraryIndependentInterpolationDecoration, we
    // cannot use this effectively in many games since no client API provides
    // interpoation qualifiers in vertex shaders.
    return m_features.extGraphicsPipelineLibrary.graphicsPipelineLibrary
        && m_properties.extGraphicsPipelineLibrary.graphicsPipelineLibraryIndependentInterpolationDecoration
        && m_options.enableGraphicsPipelineLibrary != Tristate::False;
  }


  bool DxvkDevice::canUsePipelineCacheControl() const {
    // Don't bother with this unless the device also supports shader module
    // identifiers, since decoding and hashing the shaders is slow otherwise
    // and likely provides no benefit over linking pipeline libraries.
    return m_features.vk13.pipelineCreationCacheControl
        && m_features.extShaderModuleIdentifier.shaderModuleIdentifier
        && m_options.enableGraphicsPipelineLibrary != Tristate::True;
  }


  bool DxvkDevice::mustTrackPipelineLifetime() const {
    switch (m_options.trackPipelineLifetime) {
      case Tristate::True:
        return canUseGraphicsPipelineLibrary();

      case Tristate::False:
        return false;

      default:
      case Tristate::Auto:
        if (!env::is32BitHostPlatform() || !canUseGraphicsPipelineLibrary())
          return false;

        // Disable lifetime tracking for drivers that do not have any
        // significant issues with 32-bit address space to begin with
        if (m_adapter->matchesDriver(VK_DRIVER_ID_MESA_RADV_KHR, 0, 0))
          return false;

        return true;
    }
  }


  DxvkFramebufferSize DxvkDevice::getDefaultFramebufferSize() const {
    return DxvkFramebufferSize {
      m_properties.core.properties.limits.maxFramebufferWidth,
      m_properties.core.properties.limits.maxFramebufferHeight,
      m_properties.core.properties.limits.maxFramebufferLayers };
  }


  VkPipelineStageFlags DxvkDevice::getShaderPipelineStages() const {
    VkPipelineStageFlags result = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
                                | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    
    if (m_features.core.features.geometryShader)
      result |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
    
    if (m_features.core.features.tessellationShader) {
      result |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT
             |  VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
    }

    return result;
  }


  DxvkDeviceOptions DxvkDevice::options() const {
    DxvkDeviceOptions options;
    options.maxNumDynamicUniformBuffers = m_properties.core.properties.limits.maxDescriptorSetUniformBuffersDynamic;
    options.maxNumDynamicStorageBuffers = m_properties.core.properties.limits.maxDescriptorSetStorageBuffersDynamic;
    return options;
  }
  
  
  Rc<DxvkCommandList> DxvkDevice::createCommandList() {
    Rc<DxvkCommandList> cmdList = m_recycledCommandLists.retrieveObject();
    
    if (cmdList == nullptr)
      cmdList = new DxvkCommandList(this);
    
    return cmdList;
  }


  Rc<DxvkContext> DxvkDevice::createContext(DxvkContextType type) {
    return new DxvkContext(this, type);
  }


  Rc<DxvkGpuEvent> DxvkDevice::createGpuEvent() {
    return new DxvkGpuEvent(m_vkd);
  }


  Rc<DxvkGpuQuery> DxvkDevice::createGpuQuery(
          VkQueryType           type,
          VkQueryControlFlags   flags,
          uint32_t              index) {
    return new DxvkGpuQuery(m_vkd, type, flags, index);
  }


  Rc<DxvkFence> DxvkDevice::createFence(
    const DxvkFenceCreateInfo& fenceInfo) {
    return new DxvkFence(this, fenceInfo);
  }


  Rc<DxvkBuffer> DxvkDevice::createBuffer(
    const DxvkBufferCreateInfo& createInfo,
          VkMemoryPropertyFlags memoryType) {
    return new DxvkBuffer(this, createInfo, m_objects.memoryManager(), memoryType);
  }
  
  
  Rc<DxvkBufferView> DxvkDevice::createBufferView(
    const Rc<DxvkBuffer>&           buffer,
    const DxvkBufferViewCreateInfo& createInfo) {
    return new DxvkBufferView(m_vkd, buffer, createInfo);
  }
  
  
  Rc<DxvkImage> DxvkDevice::createImage(
    const DxvkImageCreateInfo&  createInfo,
          VkMemoryPropertyFlags memoryType) {
    return new DxvkImage(this, createInfo, m_objects.memoryManager(), memoryType);
  }
  
  
  Rc<DxvkImageView> DxvkDevice::createImageView(
    const Rc<DxvkImage>&            image,
    const DxvkImageViewCreateInfo&  createInfo) {
    return new DxvkImageView(m_vkd, image, createInfo);
  }
  
  
  Rc<DxvkSampler> DxvkDevice::createSampler(
    const DxvkSamplerCreateInfo&  createInfo) {
    return new DxvkSampler(this, createInfo);
  }
  
  
  Rc<DxvkSparsePageAllocator> DxvkDevice::createSparsePageAllocator() {
    return new DxvkSparsePageAllocator(m_objects.memoryManager());
  }


  DxvkStatCounters DxvkDevice::getStatCounters() {
    DxvkPipelineCount pipe = m_objects.pipelineManager().getPipelineCount();
    DxvkPipelineWorkerStats workers = m_objects.pipelineManager().getWorkerStats();
    
    DxvkStatCounters result;
    result.setCtr(DxvkStatCounter::PipeCountGraphics, pipe.numGraphicsPipelines);
    result.setCtr(DxvkStatCounter::PipeCountLibrary,  pipe.numGraphicsLibraries);
    result.setCtr(DxvkStatCounter::PipeCountCompute,  pipe.numComputePipelines);
    result.setCtr(DxvkStatCounter::PipeTasksDone,     workers.tasksCompleted);
    result.setCtr(DxvkStatCounter::PipeTasksTotal,    workers.tasksTotal);
    result.setCtr(DxvkStatCounter::GpuIdleTicks,      m_submissionQueue.gpuIdleTicks());

    std::lock_guard<sync::Spinlock> lock(m_statLock);
    result.merge(m_statCounters);
    return result;
  }
  
  
  Rc<DxvkBuffer> DxvkDevice::importBuffer(
    const DxvkBufferCreateInfo& createInfo,
    const DxvkBufferImportInfo& importInfo,
          VkMemoryPropertyFlags memoryType) {
    return new DxvkBuffer(this, createInfo, importInfo, memoryType);
  }


  Rc<DxvkImage> DxvkDevice::importImage(
    const DxvkImageCreateInfo&  createInfo,
          VkImage               image,
          VkMemoryPropertyFlags memoryType) {
    return new DxvkImage(this, createInfo, image, memoryType);
  }


  DxvkMemoryStats DxvkDevice::getMemoryStats(uint32_t heap) {
    return m_objects.memoryManager().getMemoryStats(heap);
  }


  uint32_t DxvkDevice::getCurrentFrameId() const {
    return m_statCounters.getCtr(DxvkStatCounter::QueuePresentCount);
  }
  
  
  void DxvkDevice::registerShader(const Rc<DxvkShader>& shader) {
    if (m_shaderCache)
      m_shaderCache->onShaderCreated(shader);

    m_objects.pipelineManager().registerShader(shader);
  }


  DxvkShaderPipelineLibrary* DxvkDevice::registerShaderFromCache(const Rc<DxvkShader>& shader) {
    return m_objects.pipelineManager().registerShaderFromCache(shader);
  }


  void DxvkDevice::prewarmCachedPipelines() {
    m_objects.pipelineManager().prewarmCachedPipelines();
  }


  void DxvkDevice::prewarmShaderWithGraphics(const Rc<DxvkShader>& shader,
          DxvkShaderPipelineLibrary* library) {
    if (shader == nullptr)
      return;

    if (library != nullptr) {
      try {
        library->compilePipeline();
        Logger::ehang(str::format(
          "Prewarmed shader ", shader->debugName(),
          " via vkCreateGraphicsPipelines"));
      } catch (const DxvkError& err) {
        Logger::ehang(str::format(
          "Exception during shader prewarm for ",
          shader->debugName(), ": ", err.message()));
      }

      return;
    }

    VkShaderStageFlagBits stage = shader->info().stage;

    auto isPreRasterStage = [](VkShaderStageFlagBits s) {
      switch (s) {
        case VK_SHADER_STAGE_VERTEX_BIT:
        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
        case VK_SHADER_STAGE_GEOMETRY_BIT:
          return true;
        default:
          return false;
      }
    };

    if (!isPreRasterStage(stage) && stage != VK_SHADER_STAGE_FRAGMENT_BIT)
      return;

    if (stage == VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
      Logger::ehang(str::format(
        "Skipping tessellation control shader prewarm for ",
        shader->debugName(),
        " because no compatible stub stage is available"));
      return;
    }

    try {
      const DxvkBindingLayout& bindingLayout = shader->getBindings();

      std::array<std::unique_ptr<DxvkBindingSetLayout>, DxvkDescriptorSets::SetCount> setLayouts = { };

      std::array<const DxvkBindingSetLayout*, DxvkDescriptorSets::SetCount> setLayoutPtrs = { };

      uint32_t setCount = DxvkDescriptorSets::SetCount;

      for (uint32_t i = 0; i < setCount; i++) {
        DxvkBindingSetLayoutKey key(bindingLayout.getBindingList(i));
        setLayouts[i] = std::make_unique<DxvkBindingSetLayout>(this, key);
        setLayoutPtrs[i] = setLayouts[i].get();
      }

      DxvkBindingLayoutObjects bindingObjects(this, bindingLayout, setLayoutPtrs.data());

      SpirvCodeBuffer spirvCode = shader->getCode(&bindingObjects, DxvkShaderModuleCreateInfo());

      if (!spirvCode.size()) {
        Logger::ehang(str::format("Shader ", shader->debugName(), " has no SPIR-V data to prewarm"));
        return;
      }

      auto detectGeometryInputTopology = [](const SpirvCodeBuffer& code) {
        struct DetectionResult {
          VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
          bool found = false;
        } result;

        const uint32_t* words = code.data();
        size_t wordCount = code.dwords();

        if (wordCount < 5)
          return result;

        size_t offset = 5;

        while (offset < wordCount) {
          uint32_t word = words[offset];
          uint16_t opCode = word & 0xffffu;
          uint16_t length = word >> 16;

          if (!length)
            break;

          if (offset + length > wordCount)
            break;

          if (opCode == uint16_t(spv::OpExecutionMode) && length >= 3) {
            spv::ExecutionMode mode = spv::ExecutionMode(words[offset + 2]);

            switch (mode) {
              case spv::ExecutionModeInputPoints:
                result.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
                result.found = true;
                return result;

              case spv::ExecutionModeInputLines:
                result.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
                result.found = true;
                return result;

              case spv::ExecutionModeInputLinesAdjacency:
                result.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
                result.found = true;
                return result;

              case spv::ExecutionModeTriangles:
                result.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                result.found = true;
                return result;

              case spv::ExecutionModeInputTrianglesAdjacency:
                result.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
                result.found = true;
                return result;

              default:
                break;
            }
          }

          offset += length;
        }

        return result;
      };

      VkPrimitiveTopology inputTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

      if (stage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
        inputTopology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
      } else if (stage == VK_SHADER_STAGE_GEOMETRY_BIT) {
        auto topologyInfo = detectGeometryInputTopology(spirvCode);
        inputTopology = topologyInfo.topology;

        if (!topologyInfo.found) {
          Logger::ehang(str::format(
            "Failed to detect geometry input primitive for ",
            shader->debugName(),
            ", defaulting to triangle list"));
        }
      }

      auto vk = m_vkd;

      auto createModuleFromCode = [&](const SpirvCodeBuffer& code) -> VkShaderModule {
        if (!code.size())
          return VK_NULL_HANDLE;

        VkShaderModuleCreateInfo moduleInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        moduleInfo.codeSize = code.size();
        moduleInfo.pCode    = code.data();

        VkShaderModule module = VK_NULL_HANDLE;

        if (vk->vkCreateShaderModule(vk->device(), &moduleInfo, nullptr, &module) != VK_SUCCESS)
          return VK_NULL_HANDLE;

        return module;
      };

      std::vector<VkPipelineShaderStageCreateInfo> stageInfos;
      std::vector<VkShaderModule> ownedModules;

      auto appendStage = [&](VkShaderStageFlagBits stageFlag, VkShaderModule module) {
        VkPipelineShaderStageCreateInfo info = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        info.stage  = stageFlag;
        info.module = module;
        info.pName  = "main";
        stageInfos.push_back(info);
      };

      if (stage != VK_SHADER_STAGE_VERTEX_BIT) {
        SpirvCodeBuffer stubVertex(dxvk_fullscreen_vert);
        VkShaderModule stubModule = createModuleFromCode(stubVertex);

        if (stubModule == VK_NULL_HANDLE) {
          Logger::ehang(str::format("Failed to create stub vertex shader for prewarming ", shader->debugName()));
          return;
        }

        ownedModules.push_back(stubModule);
        appendStage(VK_SHADER_STAGE_VERTEX_BIT, stubModule);
      }

      VkShaderModule mainModule = createModuleFromCode(spirvCode);

      if (mainModule == VK_NULL_HANDLE) {
        Logger::ehang(str::format("Failed to create shader module for prewarming ", shader->debugName()));
        for (VkShaderModule module : ownedModules)
          vk->vkDestroyShaderModule(vk->device(), module, nullptr);
        return;
      }

      ownedModules.push_back(mainModule);
      appendStage(stage, mainModule);

      VkPipelineLayout pipelineLayout = bindingObjects.getPipelineLayout(false);
      bool ownsFallbackLayout = false;

      if (pipelineLayout == VK_NULL_HANDLE)
        pipelineLayout = bindingObjects.getPipelineLayout(true);

      VkPipelineLayout fallbackLayout = VK_NULL_HANDLE;

      if (pipelineLayout == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayout> descriptorLayouts(setCount);

        for (uint32_t i = 0; i < setCount; i++)
          descriptorLayouts[i] = setLayoutPtrs[i] ? setLayoutPtrs[i]->getSetLayout() : VK_NULL_HANDLE;

        VkPipelineLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = setCount;
        layoutInfo.pSetLayouts    = descriptorLayouts.data();

        VkPushConstantRange pushConst = bindingLayout.getPushConstantRange();

        if (pushConst.stageFlags && pushConst.size) {
          layoutInfo.pushConstantRangeCount = 1;
          layoutInfo.pPushConstantRanges    = &pushConst;
        }

        if (vk->vkCreatePipelineLayout(vk->device(), &layoutInfo, nullptr, &fallbackLayout) == VK_SUCCESS) {
          pipelineLayout = fallbackLayout;
          ownsFallbackLayout = true;
        } else {
          Logger::ehang(str::format("Failed to create fallback pipeline layout for prewarming ", shader->debugName()));

          for (VkShaderModule module : ownedModules)
            vk->vkDestroyShaderModule(vk->device(), module, nullptr);

          return;
        }
      }

      VkViewport viewport = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
      VkRect2D scissor = { { 0, 0 }, { 1, 1 } };

      VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
      viewportState.viewportCount = 1;
      viewportState.pViewports    = &viewport;
      viewportState.scissorCount  = 1;
      viewportState.pScissors     = &scissor;

      VkPipelineVertexInputStateCreateInfo vertexInput = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

      VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
      inputAssembly.topology = inputTopology;
      inputAssembly.primitiveRestartEnable = VK_FALSE;

      VkPipelineRasterizationStateCreateInfo rasterizer = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
      rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
      rasterizer.cullMode    = VK_CULL_MODE_NONE;
      rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      rasterizer.lineWidth   = 1.0f;

      VkPipelineMultisampleStateCreateInfo multisample = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
      multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

      VkPipelineDepthStencilStateCreateInfo depthStencil = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
      depthStencil.depthTestEnable  = VK_FALSE;
      depthStencil.depthWriteEnable = VK_FALSE;

      VkPipelineColorBlendAttachmentState colorAttachment = { };
      colorAttachment.colorWriteMask = 0xf;

      VkPipelineColorBlendStateCreateInfo colorBlend = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
      bool hasFragmentStage = stage == VK_SHADER_STAGE_FRAGMENT_BIT;
      colorBlend.attachmentCount = hasFragmentStage ? 1u : 0u;
      colorBlend.pAttachments    = hasFragmentStage ? &colorAttachment : nullptr;

      VkPipelineTessellationStateCreateInfo tessellationState = { VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO };

      if (stage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
        uint32_t patchVertices = shader->info().patchVertexCount ? shader->info().patchVertexCount : 1u;
        tessellationState.patchControlPoints = patchVertices;
      }

      VkAttachmentDescription colorDesc = { };
      colorDesc.format         = VK_FORMAT_R8G8B8A8_UNORM;
      colorDesc.samples        = VK_SAMPLE_COUNT_1_BIT;
      colorDesc.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      colorDesc.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      colorDesc.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      colorDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      colorDesc.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
      colorDesc.finalLayout    = VK_IMAGE_LAYOUT_GENERAL;

      VkAttachmentReference colorRef = { 0u, VK_IMAGE_LAYOUT_GENERAL };

      VkSubpassDescription subpass = { };
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = hasFragmentStage ? 1u : 0u;
      subpass.pColorAttachments = hasFragmentStage ? &colorRef : nullptr;

      VkRenderPassCreateInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
      renderPassInfo.attachmentCount = hasFragmentStage ? 1u : 0u;
      renderPassInfo.pAttachments    = hasFragmentStage ? &colorDesc : nullptr;
      renderPassInfo.subpassCount   = 1u;
      renderPassInfo.pSubpasses     = &subpass;

      VkRenderPass renderPass = VK_NULL_HANDLE;

      if (vk->vkCreateRenderPass(vk->device(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        Logger::ehang(str::format("Failed to create render pass for shader prewarm ", shader->debugName()));

        for (VkShaderModule module : ownedModules)
          vk->vkDestroyShaderModule(vk->device(), module, nullptr);

        if (ownsFallbackLayout)
          vk->vkDestroyPipelineLayout(vk->device(), fallbackLayout, nullptr);

        return;
      }

      VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
      pipelineInfo.stageCount          = uint32_t(stageInfos.size());
      pipelineInfo.pStages             = stageInfos.data();
      pipelineInfo.pVertexInputState   = &vertexInput;
      pipelineInfo.pInputAssemblyState = &inputAssembly;
      pipelineInfo.pViewportState      = &viewportState;
      pipelineInfo.pRasterizationState = &rasterizer;
      pipelineInfo.pMultisampleState   = &multisample;
      pipelineInfo.pDepthStencilState  = &depthStencil;
      pipelineInfo.pColorBlendState    = &colorBlend;
      pipelineInfo.layout              = pipelineLayout;
      pipelineInfo.renderPass          = renderPass;
      pipelineInfo.subpass             = 0;
      pipelineInfo.basePipelineIndex   = -1;

      if (stage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
        pipelineInfo.pTessellationState = &tessellationState;

      VkPipeline pipeline = VK_NULL_HANDLE;
      VkResult vr = vk->vkCreateGraphicsPipelines(vk->device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);

      if (vr == VK_SUCCESS) {
        Logger::ehang(str::format("Prewarmed shader ", shader->debugName(), " via vkCreateGraphicsPipelines"));
        shader->notifyLibraryCompile();
        vk->vkDestroyPipeline(vk->device(), pipeline, nullptr);
      } else {
        Logger::ehang(str::format("Failed to prewarm shader ", shader->debugName(), " via vkCreateGraphicsPipelines: ", vr));
      }

      vk->vkDestroyRenderPass(vk->device(), renderPass, nullptr);

      for (VkShaderModule module : ownedModules)
        vk->vkDestroyShaderModule(vk->device(), module, nullptr);

      if (ownsFallbackLayout)
        vk->vkDestroyPipelineLayout(vk->device(), fallbackLayout, nullptr);

    } catch (const DxvkError& err) {
      Logger::ehang(str::format("Exception during shader prewarm for ", shader->debugName(), ": ", err.message()));
    }
  }


  void DxvkDevice::prewarmShaderWithCompute(
    const Rc<DxvkShader>&         shader,
          DxvkShaderPipelineLibrary* library) {
    if (shader == nullptr)
      return;

    try {
      if (library != nullptr)
        Logger::ehang(str::format(
          "Ignoring pipeline library when prewarming compute shader ",
          shader->debugName(),
          ": compute prewarm always uses monolithic pipelines"));

      DxvkComputePipelineShaders shaders;
      shaders.cs = shader;

      auto pipeline = m_objects.pipelineManager().createComputePipeline(shaders);

      if (pipeline == nullptr) {
        Logger::ehang(str::format(
          "Failed to acquire compute pipeline for prewarm ", shader->debugName()));
        return;
      }

      DxvkComputePipelineStateInfo state;
      pipeline->compilePipeline(state);

      Logger::ehang(str::format(
        "Prewarmed compute shader ", shader->debugName(),
        " via DxvkComputePipeline::compilePipeline"));
    } catch (const DxvkError& err) {
      Logger::ehang(str::format(
        "Exception during compute shader prewarm for ",
        shader->debugName(), ": ", err.message()));
    }
  }


  void DxvkDevice::requestCompileShader(
    const Rc<DxvkShader>&           shader) {
    m_objects.pipelineManager().requestCompileShader(shader);
  }


  void DxvkDevice::presentImage(
    const Rc<Presenter>&            presenter,
          VkPresentModeKHR          presentMode,
          uint64_t                  frameId,
          DxvkSubmitStatus*         status) {
    status->result = VK_NOT_READY;

    DxvkPresentInfo presentInfo = { };
    presentInfo.presenter = presenter;
    presentInfo.presentMode = presentMode;
    presentInfo.frameId = frameId;
    m_submissionQueue.present(presentInfo, status);
    
    std::lock_guard<sync::Spinlock> statLock(m_statLock);
    m_statCounters.addCtr(DxvkStatCounter::QueuePresentCount, 1);
  }


  void DxvkDevice::submitCommandList(
    const Rc<DxvkCommandList>&      commandList,
          DxvkSubmitStatus*         status) {
    DxvkSubmitInfo submitInfo = { };
    submitInfo.cmdList = commandList;
    m_submissionQueue.submit(submitInfo, status);

    std::lock_guard<sync::Spinlock> statLock(m_statLock);
    m_statCounters.merge(commandList->statCounters());
  }
  
  
  VkResult DxvkDevice::waitForSubmission(DxvkSubmitStatus* status) {
    VkResult result = status->result.load();

    if (result == VK_NOT_READY) {
      m_submissionQueue.synchronizeSubmission(status);
      result = status->result.load();
    }

    return result;
  }


  void DxvkDevice::waitForResource(const Rc<DxvkResource>& resource, DxvkAccess access) {
    if (resource->isInUse(access)) {
      auto t0 = dxvk::high_resolution_clock::now();

      m_submissionQueue.synchronizeUntil([resource, access] {
        return !resource->isInUse(access);
      });

      auto t1 = dxvk::high_resolution_clock::now();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

      std::lock_guard<sync::Spinlock> lock(m_statLock);
      m_statCounters.addCtr(DxvkStatCounter::GpuSyncCount, 1);
      m_statCounters.addCtr(DxvkStatCounter::GpuSyncTicks, us.count());
    }
  }
  
  
  void DxvkDevice::waitForIdle() {
    m_submissionQueue.waitForIdle();
    m_submissionQueue.lockDeviceQueue();

    if (m_vkd->vkDeviceWaitIdle(m_vkd->device()) != VK_SUCCESS)
      Logger::err("DxvkDevice: waitForIdle: Operation failed");

    m_submissionQueue.unlockDeviceQueue();
  }
  
  
  DxvkDevicePerfHints DxvkDevice::getPerfHints() {
    DxvkDevicePerfHints hints;
    hints.preferFbDepthStencilCopy = m_features.extShaderStencilExport
      && (m_adapter->matchesDriver(VK_DRIVER_ID_MESA_RADV_KHR, 0, 0)
       || m_adapter->matchesDriver(VK_DRIVER_ID_AMD_OPEN_SOURCE_KHR, 0, 0)
       || m_adapter->matchesDriver(VK_DRIVER_ID_AMD_PROPRIETARY_KHR, 0, 0));
    hints.preferFbResolve = m_features.amdShaderFragmentMask
      && (m_adapter->matchesDriver(VK_DRIVER_ID_AMD_OPEN_SOURCE_KHR, 0, 0)
       || m_adapter->matchesDriver(VK_DRIVER_ID_AMD_PROPRIETARY_KHR, 0, 0));
    return hints;
  }


  void DxvkDevice::recycleCommandList(const Rc<DxvkCommandList>& cmdList) {
    m_recycledCommandLists.returnObject(cmdList);
  }
  
}
