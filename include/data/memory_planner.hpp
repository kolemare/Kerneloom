#ifndef KL_MEMORY_PLANNER_HPP
#define KL_MEMORY_PLANNER_HPP

#include <data/memory_plan.hpp>
#include <data/memory_policy.hpp>

#include <core/device.hpp>

#include <cstddef>

namespace kl
{

    MemoryPlan create_memory_plan(
        Device device,
        std::size_t batch_bytes,
        const MemoryPolicy &policy);

}

#endif // KL_MEMORY_PLANNER_HPP