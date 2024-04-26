/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/data/flat_map_utils.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {

FlatMapRandomAccessHandler::FlatMapRandomAccessHandler(
    OpKernelContext* const ctx, const DatasetBase* input_dataset,
    CapturedFunction& captured_map_func)
    : input_dataset_(input_dataset),
      captured_map_func_(captured_map_func),
      unbounded_thread_pool_(ctx->env(),
                             "tf_data_flat_map_random_access_handler") {
  absl::Status status =
      ctx->function_library()->Clone(&flib_def_, &pflr_, &flr_, true);
  if (!status.ok()) {
    cumulative_cardinalities_ = std::move(status);
    return;
  }
  function_handle_cache_ =
      std::make_unique<FunctionHandleCache>(pflr_->GetFLR("/device:CPU:0"));
  IteratorContext::Params params(ctx);
  params.cancellation_manager = &cancellation_manager_;
  params.env = ctx->env();
  params.flr = flr_;
  params.function_handle_cache = function_handle_cache_.get();
  params.resource_mgr = &resource_mgr_;
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  ctx_ = std::make_unique<IteratorContext>(std::move(params));
}

FlatMapRandomAccessHandler::~FlatMapRandomAccessHandler() {
  for (DatasetBase* dataset : input_datasets_) {
    dataset->Unref();
  }
  input_datasets_.clear();
}

absl::StatusOr<int64_t> FlatMapRandomAccessHandler::Cardinality() {
  TF_RETURN_IF_ERROR(cumulative_cardinalities_.status());
  if (cumulative_cardinalities_->empty()) {
    cumulative_cardinalities_ = ComputeCardinalities();
  }
  return cumulative_cardinalities_->back();
}

absl::StatusOr<std::vector<int64_t>>
FlatMapRandomAccessHandler::ComputeCardinalities() {
  if (input_datasets_.empty()) {
    TF_ASSIGN_OR_RETURN(input_datasets_, MakeInputDatasets());
  }

  std::vector<int64_t> cumulative_cardinalities;
  cumulative_cardinalities.reserve(input_datasets_.size());
  for (size_t i = 0; i < input_datasets_.size(); ++i) {
    int64_t input_cardinality = input_datasets_[i]->Cardinality();
    if (input_cardinality == kInfiniteCardinality ||
        input_cardinality == kUnknownCardinality) {
      cumulative_cardinalities.push_back(input_cardinality);
      return cumulative_cardinalities;
    }
    int64_t cumulative_cardinality = input_cardinality;
    if (i > 0) {
      cumulative_cardinality += cumulative_cardinalities.back();
    }
    cumulative_cardinalities.push_back(cumulative_cardinality);
  }
  if (cumulative_cardinalities.empty()) {
    cumulative_cardinalities.push_back(0);
  }
  return cumulative_cardinalities;
}

absl::StatusOr<std::vector<DatasetBase*>>
FlatMapRandomAccessHandler::MakeInputDatasets() const {
  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(input_dataset_->MakeIterator(
      ctx_.get(), /*parent=*/nullptr, "Iterator", &iterator));

  std::unique_ptr<InstantiatedCapturedFunction> instantiated_map_func;
  TF_RETURN_IF_ERROR(
      captured_map_func_.Instantiate(ctx_.get(), &instantiated_map_func));

  std::vector<DatasetBase*> input_datasets;
  while (true) {
    std::vector<Tensor> input_tensors;
    bool end_of_sequence = false;
    TF_RETURN_IF_ERROR(
        iterator->GetNext(ctx_.get(), &input_tensors, &end_of_sequence));
    if (end_of_sequence) {
      return input_datasets;
    }

    std::vector<Tensor> mapped_tensors;
    TF_RETURN_IF_ERROR(instantiated_map_func->Run(
        ctx_.get(), std::move(input_tensors), &mapped_tensors));
    if (!(mapped_tensors.size() == 1 &&
          mapped_tensors[0].dtype() == DT_VARIANT &&
          TensorShapeUtils::IsScalar(mapped_tensors[0].shape()))) {
      return absl::InvalidArgumentError(
          "Flat map function must return a single scalar of dtype DT_VARIANT "
          "representing a dataset.");
    }

    DatasetBase* mapped_dataset = nullptr;
    TF_RETURN_IF_ERROR(
        GetDatasetFromVariantTensor(mapped_tensors[0], &mapped_dataset));
    mapped_dataset->Ref();
    input_datasets.push_back(mapped_dataset);
  }
  return input_datasets;
}
}  // namespace data
}  // namespace tensorflow
