// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "contrib_ops/cpu/linalg_cholesky.h"
#include "core/framework/framework_common.h"
#include "core/framework/tensorprotoutils.h"
#include <Eigen/Dense>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
  LinalgCholesky,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
  LinalgCholesky);

#pragma warning(disable : 4189)
Status LinalgCholesky::Compute(OpKernelContext* context) const {
  Status status = Status::OK();
  const Tensor* A = context->Input<Tensor>(0);
  const TensorShape& a_shape = A->Shape();
  assert(a_shape.NumDimensions() == 2);
  assert(a_shape[0] == a_shape[1]);

  TensorShape X_shape = { a_shape[0], a_shape[1] };
  Tensor* X = context->Output(0, X_shape);

  const Eigen::StorageOptions option = Eigen::RowMajor;
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, option>> a_matrix(A->Data<float>(), narrow<size_t>(a_shape[0]), narrow<size_t>(a_shape[1]));
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, option>> x_matrix(X->MutableData<float>(), narrow<size_t>(a_shape[0]), narrow<size_t>(a_shape[1]));
  Eigen::LLT<Eigen::MatrixXf> lltOfA(a_matrix);
  if (lltOfA.info() == Eigen::Success) {
    if (this->upper_) {
      x_matrix = lltOfA.matrixU();
    }
    else {
      x_matrix = lltOfA.matrixL();
    }
  }
  else {
    assert(false);
  }
  return status;
}
}  // namespace contrib
}  // namespace onnxruntime
