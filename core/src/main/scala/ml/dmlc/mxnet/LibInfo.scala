package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

import scala.collection.mutable.ListBuffer

class LibInfo {
  @native def mxNDArrayFree(handle: NDArrayHandle): Int
  @native def mxGetLastError(): String
  @native def mxNDArrayCreateNone(out: NDArrayHandle): Int
  @native def mxNDArrayCreate(shape: Array[Int], ndim: Int, devType: Int,
                              devId: Int, delayAlloc: Int, out: NDArrayHandle): Int
  @native def mxNDArrayWaitAll(): Int
  @native def mxListFunctions(functions: ListBuffer[FunctionHandle]): Int
}
