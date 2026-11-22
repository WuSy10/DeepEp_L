/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include "common/types.h"
#include "runtime/kernel_trace.h"
#include "runtime/shmem_instance.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using ascend_deepep::trace::KernelTrace;
    using ascend_deepep::trace::TraceRecord;
    auto to_dict = [](const TraceRecord& record) {
        pybind11::dict result;
        result["event"] = record.event;
        if (!record.op_name.empty()) result["op_name"] = record.op_name;
        if (!record.kernel_name.empty()) result["kernel_name"] = record.kernel_name;
        for (const auto& item : record.values) result[pybind11::str(item.first)] = item.second;
        return result;
    };
    pybind11::class_<KernelTrace, std::shared_ptr<KernelTrace>>(m, "KernelTrace")
        .def_property_readonly("identity", [](const KernelTrace& trace) {
            return reinterpret_cast<uintptr_t>(&trace);
        })
        .def("calibration", [to_dict](const KernelTrace& trace) { return to_dict(trace.Calibration()); })
        .def("drain_trace", [to_dict](KernelTrace& trace, bool closing) {
            std::vector<TraceRecord> records;
            {
                pybind11::gil_scoped_release release;
                records = trace.Drain(closing);
            }
            pybind11::list result;
            for (const auto& record : records) result.append(to_dict(record));
            return result;
        }, pybind11::arg("closing") = false)
        .def("synchronize", &KernelTrace::Synchronize, pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("get_kernel_trace", &KernelTrace::Open);
    m.def("get_shmem_unique_id" , &GetShmemUniqueId);
    m.def("get_shmem_bootstrap", &GetShmemBootstrap);
    // Keep the Python dtype contract tied to the native V1 dispatch path.
    m.attr("topk_idx_t") = pybind11::cast(c10::CppTypeToScalarType<ascend_deepep::topk_idx_t>::value);
}
