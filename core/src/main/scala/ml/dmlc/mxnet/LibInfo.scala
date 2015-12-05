package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

class LibInfo {
  @native def mxNDArrayFree(handle: NDArrayHandle): Int
  @native def mxGetLastError(): String
  @native def mxNDArrayCreateNone(out: NDArrayHandle): Int
}
