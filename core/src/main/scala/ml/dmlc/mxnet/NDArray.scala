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

  val functions: Map[String, NDArrayFunction] = _initNdarrayModule()

  // Definition of internal functions.
  // Internal binary function
  def binaryNdarrayFunction(funcName: String, lhs: NDArray, rhs: NDArray, out: NDArray = null): NDArray = {
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
  def unaryNDArrayFunction(funcName: String, src: NDArray, out: NDArray = null): NDArray = {
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
  def genericNDArrayFunction(funcName: String,
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
    Return a new empty handle.

    Empty handle can be used to hold result

    Returns
    -------
    a new empty ndarray handle
  */
  def _newEmptyHandle(): NDArrayHandle = {
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
  def _newAllocHandle(shape: Array[Int], ctx: Context, delayAlloc: Boolean): NDArrayHandle = {
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
    Wait all async operation to finish in MXNet

    This function is used for benchmark only
  */
  def waitall(): Unit = {
    checkCall(_LIB.mxNDArrayWaitAll())
  }

  // Create a NDArray function from the FunctionHandle.
  def _makeNdarrayFunction(handle: FunctionHandle): (String, NDArrayFunction) = {
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
    val paramStr = ctypes2docstring(argNames, argTypes, argDescs)
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n"
    println(docStr)
    if (nMutateVars.value == 1 && nUsedVars.value == 2 && nScalars.value == 0) {
      (name.value, BinaryNDArrayFunction(handle, acceptEmptyMutate))
    } else if (nMutateVars.value == 1 && nUsedVars.value == 1 && nScalars.value == 0) {
      (name.value, UnaryNDArrayFunction(handle, acceptEmptyMutate))
    } else {
      (name.value, GenericNDArrayFunction(handle, acceptEmptyMutate, nMutateVars.value, useVarsRange, scalarRange))
    }
  }

  // List and add all the ndarray functions to current module.
  def _initNdarrayModule(): Map[String, NDArrayFunction] = {
    val functions = ListBuffer[FunctionHandle]()
    checkCall(_LIB.mxListFunctions(functions))
    functions.map(_makeNdarrayFunction).toMap
  }

  def main(args: Array[String]): Unit = {
    println("NDArray (empty) address:")
    val ndArrayEmpty: NDArrayHandle = _newEmptyHandle()
    println(ndArrayEmpty.value)

    println("NDArray (cpu) address:")
    val ctx = new Context("cpu", 0)
    val ndArrayCpu = _newAllocHandle(Array(2, 1), ctx, false)
    println(ndArrayCpu.value)

    val array1 = new NDArray(_newAllocHandle(Array(2, 1), ctx, false))
    val array2 = new NDArray(ndArrayCpu)
    println(s"Shape: ${array1.shape().mkString(",")}")
    println(s"Array1: [${array1.toArray().mkString(",")}]")
    println(s"Array2: [${array2.toArray().mkString(",")}]")
    array1 += array2
    println(s"Array1 after plus: [${array1.toArray().mkString(",")}]")
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
    NDArray.binaryNdarrayFunction("_plus", this, other)
  }

  def +(other: Double): NDArray = {
    NDArray._plusScalar(this, other)
  }

  def +=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray.binaryNdarrayFunction("_plus", this, other, out=this)
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
  def toArray(): Array[MXFloat] = {
    val size = shape().product
    val data = Array.ofDim[MXFloat](size)
    checkCall(_LIB.mxNDArraySyncCopyToCPU(handle, data, size))
    data
  }

  /**
    Get shape of current NDArray.

    Returns
    -------
    a tuple representing shape of current ndarray
  */
  def shape(): Array[Int] = {
    val ndim = new MXUintRef
    val data = ArrayBuffer[Int]()
    checkCall(_LIB.mxNDArrayGetShape(handle, ndim, data))
    require(ndim.value == data.length, s"ndim=$ndim, while len(pdata)=${data.length}")
    data.toArray
  }
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
    NDArray._rminusScalar(other, value.asInstanceOf[Double])
  }

  def *(other: NDArray): NDArray = {
    other * value.asInstanceOf[Double]
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
