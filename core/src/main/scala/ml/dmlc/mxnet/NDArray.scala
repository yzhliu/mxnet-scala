package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

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

  private val functions: Map[String, NDArrayFunction] = _initNdarrayModule()

  // Definition of internal functions.
  // Internal binary function
  private def binaryNdarrayFunction(funcName: String, lhs: NDArray, rhs: NDArray, out: NDArray = null): NDArray = {
    var output = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    require(output == null || output.writable, "out must be writable")
    function match {
      case BinaryNDArrayFunction(handle: NDArrayHandle, acceptEmptyMutate: Boolean) =>
        if (output == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          output = new NDArray(_newEmptyHandle())
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          Array(lhs.handle.value, rhs.handle.value),
          Array[MXFloat](),
          Array(output.handle.value)))
      case _ => throw new RuntimeException(s"call $funcName as binary function")
    }
    output
  }

  // internal NDArray function
  private def unaryNDArrayFunction(funcName: String, src: NDArray, out: NDArray = null): NDArray = {
    var output = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    require(output == null || output.writable, "out must be writable")
    function match {
      case UnaryNDArrayFunction(handle: NDArrayHandle, acceptEmptyMutate: Boolean) =>
        if (output == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          output = new NDArray(_newEmptyHandle())
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          Array(src.handle.value),
          Array[MXFloat](),
          Array(output.handle.value)))
      case _ => throw new RuntimeException(s"call $funcName as unary function")
    }
    output
  }

  /**
   * Invoke this function by passing in parameters
   *
   * @param args Positional arguments of input scalars and NDArray
   * @param out NDArray or tuple of NDArray, optional
   *            Output NDArray, used to hold the output result.
   * @return The result NDArray(tuple) of result of computation.
   */
  private def genericNDArrayFunction(funcName: String,
                                     args: Array[Any],
                                     out: Array[NDArray] = null): Array[NDArray] = {
    var mutateVars = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    function match {
      case GenericNDArrayFunction(handle: FunctionHandle,
                                  acceptEmptyMutate: Boolean,
                                  nMutateVars: Int,
                                  useVarsRange: Range,
                                  scalarRange: Range) =>
        require(mutateVars == null || nMutateVars == mutateVars.length,
          s"expect $nMutateVars in $funcName")
        if (mutateVars == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          mutateVars = Array.fill[NDArray](nMutateVars)(new NDArray(_newEmptyHandle()))
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          useVarsRange.map(args(_).asInstanceOf[NDArray].handle.value).toArray,
          scalarRange.map(args(_).asInstanceOf[MXFloat]).toArray,
          mutateVars.map(_.handle.value).array))
      case _ => throw new RuntimeException(s"call $funcName as generic function")
    }
    mutateVars
  }

  /**
   * Return a new empty handle.
   * Empty handle can be used to hold result
   *
   * @return a new empty ndarray handle
   */
  private def _newEmptyHandle(): NDArrayHandle = {
    val hdl: NDArrayHandle = new NDArrayHandle
    checkCall(_LIB.mxNDArrayCreateNone(hdl))
    hdl
  }

  /**
   * Return a new handle with specified shape and context.
   * Empty handle is only used to hold results
   *
   * @return a new empty ndarray handle
   */
  private def _newAllocHandle(shape: Array[Int], ctx: Context, delayAlloc: Boolean): NDArrayHandle = {
    val hdl = new NDArrayHandle
    checkCall(_LIB.mxNDArrayCreate(
      shape,
      shape.length,
      ctx.deviceTypeid,
      ctx.deviceId,
      if (delayAlloc) 1 else 0,
      hdl))
    hdl
  }

  /**
   * Wait all async operation to finish in MXNet
   * This function is used for benchmark only
   */
  def waitall(): Unit = {
    checkCall(_LIB.mxNDArrayWaitAll())
  }

  // Create a NDArray function from the FunctionHandle.
  private def _makeNdarrayFunction(handle: FunctionHandle): (String, NDArrayFunction) = {
    val NDARRAY_ARG_BEFORE_SCALAR = 1
    val ACCEPT_EMPTY_MUTATE_TARGET = 1 << 2
    // Get the property of NDArray
    val nUsedVars = new MXUintRef
    val nScalars = new MXUintRef
    val nMutateVars = new MXUintRef
    val typeMask = new RefInt
    checkCall(_LIB.mxFuncDescribe(handle, nUsedVars, nScalars, nMutateVars, typeMask))
    val acceptEmptyMutate = (typeMask.value & ACCEPT_EMPTY_MUTATE_TARGET) != 0
    // infer type of the function
    val ndarrayArgBeforeScalar = (typeMask.value & NDARRAY_ARG_BEFORE_SCALAR) != 0
    val useVarsRange: Range =
      if (ndarrayArgBeforeScalar) 0 until nUsedVars.value
      else nScalars.value until (nUsedVars.value + nScalars.value)
    val scalarRange: Range =
      if (ndarrayArgBeforeScalar) nUsedVars.value until (nUsedVars.value + nScalars.value)
      else 0 until nScalars.value
    // Get the information from the function
    val name = new RefString
    val desc = new RefString
    val numArgs = new MXUintRef
    val argNames = ListBuffer[String]()
    val argTypes = ListBuffer[String]()
    val argDescs = ListBuffer[String]()

    checkCall(_LIB.mxFuncGetInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs))
    if (nMutateVars.value == 1 && nUsedVars.value == 2 && nScalars.value == 0) {
      (name.value, BinaryNDArrayFunction(handle, acceptEmptyMutate))
    } else if (nMutateVars.value == 1 && nUsedVars.value == 1 && nScalars.value == 0) {
      (name.value, UnaryNDArrayFunction(handle, acceptEmptyMutate))
    } else {
      (name.value, GenericNDArrayFunction(handle, acceptEmptyMutate, nMutateVars.value, useVarsRange, scalarRange))
    }
  }

  // List and add all the ndarray functions to current module.
  private def _initNdarrayModule(): Map[String, NDArrayFunction] = {
    val functions = ListBuffer[FunctionHandle]()
    checkCall(_LIB.mxListFunctions(functions))
    functions.map(_makeNdarrayFunction).toMap
  }

  /**
   * Create an empty uninitialized new NDArray, with specified shape.
   *
   * @param shape shape of the NDArray.
   * @param ctx The context of the NDArray, default to current default context.
   *
   * @return The created NDArray.
   */
  def empty(shape: Array[Int], ctx: Context=null): NDArray = {
    val context = if (ctx == null) Context.defaultCtx else ctx
    new NDArray(handle = NDArray._newAllocHandle(shape, context, delayAlloc = false))
  }

  /**
   * Create a new NDArray filled with 0, with specified shape.
   *
   * @param shape shape of the NDArray.
   * @param ctx The context of the NDArray, default to current default context.
   *
   * @return The created NDArray.
   */
  def zeros(shape: Array[Int], ctx: Context=null): NDArray = {
    val arr = empty(shape, ctx)
    arr(0).set(0f)
    arr
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

  /**
   * Return a sliced NDArray that shares memory with current one.
   * NDArray only support continuous slicing on axis 0
   *
   * @param start Starting index of slice.
   * @param stop Finishing index of slice.
   *
   * @return a sliced NDArray that shares memory with current one.
   */
  private def _slice(start: Int, stop: Int): NDArray = {
    val sliceHandle = new NDArrayHandle()
    checkCall(_LIB.mxNDArraySlice(handle, start, stop, sliceHandle))
    new NDArray(handle = sliceHandle, writable = this.writable)
  }

  private def _slice(start: Int): NDArray = {
    _slice(start, shape(0))
  }

  def apply(sliceStart: Int): NDArray = _slice(sliceStart)
  def apply(sliceStart: Int, sliceEnd: Int): NDArray = _slice(sliceStart, sliceEnd)

  /**
   * Set the values of the NDArray
   * @param value Value to set
   * @return Current NDArray
   */
  def set(value: Float): NDArray = {
    require(writable, "trying to assign to a readonly NDArray")
    NDArray.genericNDArrayFunction("_set_value", Array[Any](value), out=Array(this))
    this
  }

  def set(other: NDArray) = ???

  def +(other: NDArray): NDArray = {
    NDArray.binaryNdarrayFunction("_plus", this, other)
  }

  def +(other: Float): NDArray = {
    NDArray._plusScalar(this, other)
  }

  def +=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray.binaryNdarrayFunction("_plus", this, other, out=this)
  }

  def +=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray._plusScalar(this, other, out=this)
  }

  def -(other: NDArray): NDArray = {
    NDArray._minus(this, other)
  }

  def -(other: Float): NDArray = {
    NDArray._minusScalar(this, other)
  }

  def -=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray._minus(this, other, out=this)
  }

  def -=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray._minusScalar(this, other, out=this)
  }

  def *(other: NDArray) = {
    NDArray._mul(this, other)
  }

  def *(other: Float) = {
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

  def *=(other: Float) = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray._mulScalar(this, other, out=this)
  }

  def /(other: NDArray): NDArray = {
    NDArray._div(this, other)
  }

  def /(other: Float): NDArray = {
    NDArray._divScalar(this, other)
  }

  def /=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray._div(this, other, out=this)
  }

  def /=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray._divScalar(this, other, out=this)
  }

  /**
   * Return a copied flat java array of current array.
   * @return  A copy of array content.
   */
  // TODO: Shall we use column-major or row-major ?
  def toArray: Array[Float] = {
    val data = Array.ofDim[Float](size)
    checkCall(_LIB.mxNDArraySyncCopyToCPU(handle, data, size))
    data
  }

  /**
   * Get shape of current NDArray.
   * @return an array representing shape of current ndarray
   */
  def shape: Array[Int] = {
    val ndim = new MXUintRef
    val data = ArrayBuffer[Int]()
    checkCall(_LIB.mxNDArrayGetShape(handle, ndim, data))
    require(ndim.value == data.length, s"ndim=$ndim, while len(pdata)=${data.length}")
    data.toArray
  }

  // Get size of current NDArray.
  def size: Int = shape.product
}

object NDArrayConversions {
  implicit def int2Scalar(x: Int): NDArrayConversions[Int] = new NDArrayConversions(x)
  implicit def double2Scalar(x: Double): NDArrayConversions[Double] = new NDArrayConversions(x)
  implicit def float2Scalar(x: Float): NDArrayConversions[Float] = new NDArrayConversions(x)
}

class NDArrayConversions[@specialized(Int, Float, Double) V](val value: V) {
  def +(other: NDArray): NDArray = {
    other + value.asInstanceOf[Float]
  }

  def -(other: NDArray): NDArray = {
    NDArray._rminusScalar(other, value.asInstanceOf[Double])
  }

  def *(other: NDArray): NDArray = {
    other * value.asInstanceOf[Float]
  }

  def /(other: NDArray): NDArray = {
    NDArray._rdivScalar(other, value.asInstanceOf[Double])
  }
}

sealed class NDArrayFunction
case class BinaryNDArrayFunction(handle: FunctionHandle,
                                 acceptEmptyMutate: Boolean) extends NDArrayFunction
case class UnaryNDArrayFunction(handle: FunctionHandle,
                                acceptEmptyMutate: Boolean) extends NDArrayFunction
case class GenericNDArrayFunction(handle: FunctionHandle,
                                  acceptEmptyMutate: Boolean,
                                  nMutateVars: Int,
                                  useVarsRange: Range,
                                  scalarRange: Range) extends NDArrayFunction
