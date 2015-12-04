package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

/**
 * NDArray object in mxnet.
 * NDArray is basic ndarray/Tensor like data structure in mxnet.
 */
class NDArray(val handle: NDArrayHandle, val writable: Boolean = true) {
  override def finalize() = {
    _LIB.mxNDArrayFree(handle)
    check_call(_LIB.mxNDArrayFree(handle))
  }
}
