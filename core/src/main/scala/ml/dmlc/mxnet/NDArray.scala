package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.LoggerFactory

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object NDArray {
  private val logger = LoggerFactory.getLogger(classOf[NDArray]);

  def main(s: Array[String]) = {
    val nd = NDArray.zeros(Array(1, 2))
    import NDArrayConversions._
    val nd1 = 1 - nd
    println(nd1.toArray.mkString(","))
    println((2 * nd1).toArray.mkString(","))
    println((3 / (2 * nd1)).toArray.mkString(","))
  }

  private val functions: Map[String, NDArrayFunction] = _initNdarrayModule()

  // Definition of internal functions.
  // Internal binary function
  private[mxnet] def binaryNdarrayFunction(funcName: String,
                                           lhs: NDArray,
                                           rhs: NDArray,
                                           out: NDArray = null): NDArray = {
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
  private[mxnet] def unaryNDArrayFunction(funcName: String, src: NDArray, out: NDArray = null): NDArray = {
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
  private[mxnet] def genericNDArrayFunction(funcName: String,
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
    val paramStr = Base.ctypes2docstring(argNames, argTypes, argDescs)
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n"
    logger.debug("NDArray function defination:\n{}", docStr)
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
    NDArray.genericNDArrayFunction("_plus_scalar", Array[Any](this, other))(0)
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
    NDArray.genericNDArrayFunction("_plus_scalar", Array[Any](this, other), out=Array(this))
    this
  }

  def -(other: NDArray): NDArray = {
    NDArray.binaryNdarrayFunction("_minus", this, other)
  }

  def -(other: Float): NDArray = {
    NDArray.genericNDArrayFunction("_minus_scalar", Array[Any](this, other))(0)
  }

  def -=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray.binaryNdarrayFunction("_minus", this, other, out=this)
  }

  def -=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray.genericNDArrayFunction("_minus_scalar", Array[Any](this, other), out=Array(this))
    this
  }

  def *(other: NDArray): NDArray = {
    NDArray.binaryNdarrayFunction("_mul", this, other)
  }

  def *(other: Float): NDArray = {
    NDArray.genericNDArrayFunction("_mul_scalar", Array[Any](this, other))(0)
  }

  def unary_-(): NDArray = {
    NDArray.genericNDArrayFunction("_mul_scalar", Array[Any](this, -1f))(0)
  }

  def *=(other: NDArray) = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray.binaryNdarrayFunction("_mul", this, other, out=this)
  }

  def *=(other: Float) = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray.genericNDArrayFunction("_mul_scalar", Array[Any](this, other), out=Array(this))
    this
  }

  def /(other: NDArray): NDArray = {
    NDArray.binaryNdarrayFunction("_div", this, other)
  }

  def /(other: Float): NDArray = {
    NDArray.genericNDArrayFunction("_div_scalar", Array[Any](this, other))(0)
  }

  def /=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray.binaryNdarrayFunction("_div", this, other, out=this)
  }

  def /=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray.genericNDArrayFunction("_div_scalar", Array[Any](this, other), out=Array(this))
    this
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
  implicit def int2Scalar(x: Int): NDArrayConversions = new NDArrayConversions(x.toFloat)
  implicit def double2Scalar(x: Double): NDArrayConversions = new NDArrayConversions(x.toFloat)
  implicit def float2Scalar(x: Float): NDArrayConversions = new NDArrayConversions(x)
}

class NDArrayConversions(val value: Float) {
  def +(other: NDArray): NDArray = {
    other + value
  }

  def -(other: NDArray): NDArray = {
    NDArray.genericNDArrayFunction("_rminus_scalar", Array[Any](other, value))(0)
  }

  def *(other: NDArray): NDArray = {
    other * value
  }

  def /(other: NDArray): NDArray = {
    NDArray.genericNDArrayFunction("_rdiv_scalar", Array[Any](other, value))(0)
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
