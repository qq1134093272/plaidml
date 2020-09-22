// Vulkan device implementation, originally from the LLVM project, and
// subsequently modified by Intel Corporation.
//
// Original copyright:
//
//===- VulkanDevice.cpp - MLIR Vulkan device ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pmlc/rt/vulkan/vulkan_device.h"

#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pmlc/rt/vulkan/vulkan_error.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::rt::vulkan {

VulkanDevice::VulkanDevice(const VkPhysicalDevice &physicalDevice,
                           std::shared_ptr<VulkanState> state)
    : state{std::move(state)} {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  IVLOG(1, "Instantiating Vulkan device: " << props.deviceName);

  timestampPeriod = props.limits.timestampPeriod;

  getBestComputeQueue(physicalDevice);

  const float queuePrioritory = 1.0f;
  VkDeviceQueueCreateInfo deviceQueueCreateInfo = {};
  deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  deviceQueueCreateInfo.pNext = nullptr;
  deviceQueueCreateInfo.flags = 0;
  deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
  deviceQueueCreateInfo.queueCount = 1;
  deviceQueueCreateInfo.pQueuePriorities = &queuePrioritory;

  // Structure specifying parameters of a newly created device.
  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.pNext = nullptr;
  deviceCreateInfo.flags = 0;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
  deviceCreateInfo.enabledLayerCount = 0;
  deviceCreateInfo.ppEnabledLayerNames = nullptr;
  deviceCreateInfo.enabledExtensionCount = 0;
  deviceCreateInfo.ppEnabledExtensionNames = nullptr;
  deviceCreateInfo.pEnabledFeatures = nullptr;

  throwOnVulkanError(
      vkCreateDevice(physicalDevice, &deviceCreateInfo, 0, &device),
      "vkCreateDevice");

  VkPhysicalDeviceMemoryProperties properties = {};
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);

  // Try to find memory type with following properties:
  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT bit specifies that memory allocated
  // with this type can be mapped for host access using vkMapMemory;
  // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT bit specifies that the host cache
  // management commands vkFlushMappedMemoryRanges and
  // vkInvalidateMappedMemoryRanges are not needed to flush host writes to the
  // device or make device writes visible to the host, respectively.
  for (uint32_t i = 0, e = properties.memoryTypeCount; i < e; ++i) {
    if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT &
         properties.memoryTypes[i].propertyFlags) &&
        (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT &
         properties.memoryTypes[i].propertyFlags)) {
      memoryTypeIndex = i;
      break;
    }
  }

  if (memoryTypeIndex == VK_MAX_MEMORY_TYPES) {
    throw std::runtime_error{"invalid memoryTypeIndex"};
  }

  // Get working queue.
  vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
}

VulkanDevice::~VulkanDevice() { vkDestroyDevice(device, nullptr); }

void VulkanDevice::getBestComputeQueue(const VkPhysicalDevice &physicalDevice) {
  uint32_t queueFamilyPropertiesCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice,
                                           &queueFamilyPropertiesCount, 0);
  SmallVector<VkQueueFamilyProperties, 1> queueFamilyProperties(
      queueFamilyPropertiesCount);

  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice,
                                           &queueFamilyPropertiesCount,
                                           queueFamilyProperties.data());

  // By default a max heap is created ordered by first element of pair.
  std::priority_queue<std::pair<int, int>> queueFamilyIndexes;

  // Queue family priorities (larger numbers for higher priority)
  // 4: supports compute operations and timestamps, does not support graphics
  // operations 3: supports compute and graphics operations and timestamps 2:
  // supports compute operations, does not supports timestamps nor graphics
  // operations 1: supports compute and graphics operations, does not support
  // timestamps 0: does not support compute operations

  for (uint32_t i = 0; i < queueFamilyPropertiesCount; ++i) {
    const VkQueueFlags maskedFlags =
        (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
         queueFamilyProperties[i].queueFlags);
    int queueFamilyPriority = 0;
    if (VK_QUEUE_COMPUTE_BIT & maskedFlags) {
      queueFamilyPriority++;
      if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags)) {
        queueFamilyPriority++;
      }
      if (queueFamilyProperties[i].timestampValidBits > 0) {
        queueFamilyPriority += 2;
      }
      if (queueFamilyPriority == 4) {
        queueFamilyIndex = i;
        timestampValidBits = queueFamilyProperties[i].timestampValidBits;
        return;
      }
      queueFamilyIndexes.push(std::make_pair(queueFamilyPriority, i));
    }
  }

  if (queueFamilyIndexes.empty()) {
    throw std::runtime_error{"Cannot find a valid queue"};
  }
  queueFamilyIndex = queueFamilyIndexes.top().second;
  timestampValidBits =
      queueFamilyProperties[queueFamilyIndex].timestampValidBits;
  if (timestampValidBits > 0) {
    IVLOG(1, "  Selected device queue family supports " << timestampValidBits
                                                        << "-bit timestamps");
  } else {
    IVLOG(1, "  Selected device queue family does not supports timestamps");
  }
}

} // namespace pmlc::rt::vulkan
