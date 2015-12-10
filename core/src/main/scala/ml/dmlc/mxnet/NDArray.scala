package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

import scala.collection.mutable.ListBuffer

object NDArray {
  def _plus(array1: NDArray, array2: NDArray, out: NDArray = null): NDArray = ???
  def _plusScalar(array: NDArray, number: Double, out: NDArray = null): NDArray = ???
  def _minus(array1: NDArray, array2: NDArray, out: NDArray = null): NDArray = ???
  def _minusScalar(array: NDArray, number: Double, out: NDArray = null): NDArray = ???
  def _rminusScalar(array: NDArray, number: Double, out: NDArray = null): NDArray = ???
  def _mul(array1: NDArray, array2: NDArray, out: NDArray = null): NDArray = ???
  def _mulScalar(array: NDArray, number: Double, out: NDArray = null): NDArray = ???
  def _div(array1: NDArray, array2: NDArray, out: NDArray = null): NDArray = ???
  def _divScalar(array: NDArray, number: Double, out: NDArray = null): NDArray = ???
  def _rdivScalar(array: NDArray, number: Double, out: NDArray = null): NDArray = ???

  /**
    Return a new empty handle.

    Empty handle can be used to hold result

    Returns
    -------
    a new empty ndarray handle
  */
  def _new_empty_handle(): NDArrayHandle = {
    val hdl: NDArrayHandle = new NDArrayHandle
    checkCall(_LIB.mxNDArrayCreateNone(hdl))
    hdl
  }

  /**
    Return a new handle with specified shape and context.

    Empty handle is only used to hold results

    Returns
    -------
    a new empty ndarray handle
  */
  def _new_alloc_handle(shape: Vector[Int], ctx: Context, delayAlloc: Boolean): NDArrayHandle = {
    val hdl = new NDArrayHandle
    checkCall(_LIB.mxNDArrayCreate(
      shape.toArray,
      shape.size,
      ctx.deviceTypeid,
      ctx.deviceId,
      if (delayAlloc) 1 else 0,
      hdl))
    hdl
  }

  /**
    Wait all async operation to finish in MXNet

    This function is used for benchmark only
  */
  def waitall(): Unit = {
    checkCall(_LIB.mxNDArrayWaitAll())
  }

  // List and add all the ndarray functions to current module.
  def _init_ndarray_module(): Unit = {
    val functions: ListBuffer[FunctionHandle] = ListBuffer()
    checkCall(_LIB.mxListFunctions(functions))

    println("Functions: ")
    println(functions.length)
    functions.foreach(function => println(function.ptr64))

    /*
    module_obj = sys.modules[__name__]
    for i in range(size.value):
      hdl = FunctionHandle(plist[i])
    function = _make_ndarray_function(hdl)#if function name starts with underscore, register as static method of NDArray
    if function.__name__.startswith('_'):
      setattr (NDArray, function.__name__, staticmethod(function))
    else:
      setattr(module_obj, function.__name__, function)
      */
  }

  def main(args: Array[String]): Unit = {
    println("NDArray (empty) address:")
    val ndArrayEmpty: NDArrayHandle = _new_empty_handle()
    println(ndArrayEmpty.ptr64)

    println("NDArray (cpu) address:")
    val ctx = new Context("cpu", 0)
    val ndArrayCpu = _new_alloc_handle(Vector(2, 1), ctx, false)
    println(ndArrayCpu.ptr64)

    _init_ndarray_module()
  }
}

/**
 * NDArray object in mxnet.
 * NDArray is basic ndarray/Tensor like data structure in mxnet.
 */
class NDArray(val handle: NDArrayHandle, val writable: Boolean = true) {
  override def finalize() = {
    checkCall(_LIB.mxNDArrayFree(handle))
  }

  def +(other: NDArray): NDArray = {
    NDArray._plus(this, other)
  }

  def +(other: Double): NDArray = {
    NDArray._plusScalar(this, other)
  }

  def +=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray._plus(this, other, out=this)
  }

  def +=(other: Double): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray._plusScalar(this, other, out=this)
  }

  def -(other: NDArray): NDArray = {
    NDArray._minus(this, other)
  }

  def -(other: Double): NDArray = {
    NDArray._minusScalar(this, other)
  }

  def -=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray._minus(this, other, out=this)
  }

  def -=(other: Double): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray._minusScalar(this, other, out=this)
  }

  def *(other: NDArray) = {
    NDArray._mul(this, other)
  }

  def *(other: Double) = {
    NDArray._mulScalar(this, other)
  }

  def unary_-(): NDArray = {
    NDArray._mulScalar(this, -1.0)
  }

  def *=(other: NDArray) = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray._mul(this, other, out=this)
  }

  def *=(other: Double) = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray._mulScalar(this, other, out=this)
  }

  def /(other: NDArray): NDArray = {
    NDArray._div(this, other)
  }

  def /(other: Double): NDArray = {
    NDArray._divScalar(this, other)
  }

  def /=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray._div(this, other, out=this)
  }

  def /=(other: Double): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray._divScalar(this, other, out=this)
  }

  /**
    Return a copied numpy array of current array.

    Returns
    -------
    array : numpy.ndarray
        A copy of array content.
  */
  def asArray(): Array[Array[Double]] = {
    // TODO
    val (nRows, nCols) = shape()
    val data = Array.ofDim[Double](nRows, nCols)
    /* TODO
    checkCall(_LIB.mxNDArraySyncCopyToCPU(
      self.handle,
      data.ctypes.data_as(mx_float_p),
      ctypes.c_size_t(data.size)))
      */
    data
  }

  /**
    Get shape of current NDArray.

    Returns
    -------
    a tuple representing shape of current ndarray
  */
  def shape(): (Int, Int) = ???
}

object NDArrayConversions {
  implicit def int2Scalar(x: Int): NDArrayConversions[Int] = new NDArrayConversions(x)
  implicit def double2Scalar(x: Double): NDArrayConversions[Double] = new NDArrayConversions(x)
  implicit def float2Scalar(x: Float): NDArrayConversions[Float] = new NDArrayConversions(x)
}

class NDArrayConversions[@specialized(Int, Float, Double) V](val value: V) {
  def +(other: NDArray): NDArray = {
    other + value.asInstanceOf[Double]
  }

  def -(other: NDArray): NDArray = {
    other - value.asInstanceOf[Double]
    NDArray._rminusScalar(other, value.asInstanceOf[Double])
  }

  def *(other: NDArray): NDArray = {
    other * value.asInstanceOf[Double]
  }

  def /(other: NDArray): NDArray = {
    NDArray._rdivScalar(other, value.asInstanceOf[Double])
  }
}
