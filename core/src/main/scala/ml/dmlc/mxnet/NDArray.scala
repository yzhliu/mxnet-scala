package ml.dmlc.mxnet

/**
 * NDArray object in mxnet.
 * NDArray is basic ndarray/Tensor like data structure in mxnet.
 */
class NDArray(val handle: NDArrayHandle, val writable: Boolean = true) {
  override def finalize() = {
    /* TODO
    check_call(_LIB.MXNDArrayFree(self.handle))
    */
  }
}
