package ml.dmlc.mxnet

// type definitions
/*
mx_float = ctypes.c_float
mx_float_p = ctypes.POINTER(mx_float)
NDArrayHandle = ctypes.c_void_p
FunctionHandle = ctypes.c_void_p
SymbolCreatorHandle = ctypes.c_void_p
SymbolHandle = ctypes.c_void_p
ExecutorHandle = ctypes.c_void_p
DataIterCreatorHandle = ctypes.c_void_p
DataIterHandle = ctypes.c_void_p
KVStoreHandle = ctypes.c_void_p
RecordIOHandle = ctypes.c_void_p
RtcHandle = ctypes.c_void_p
*/

object Base {
  class RefInt(val value: Int = 0)
  class RefLong(val value: Long = 0)
  class RefString(val value: String = null)

  type MXUintRef = RefInt
  type NDArrayHandle = RefLong
  type FunctionHandle = RefLong

  // TODO
  System.loadLibrary("mxnet-scala")
  val _LIB = new LibInfo


  // helper function definition
  /**
   * Check the return value of C API call
   *
   * This function will raise exception when error occurs.
   * Wrap every API call with this function
   * Parameters
   * ----------
   * @return value from API calls
   */
  def checkCall(ret: Int): Unit = {
    if (ret != 0) {
      throw new MXNetError(_LIB.mxGetLastError())
    }
  }

  // Convert ctypes returned doc string information into parameters docstring.
  def ctypes2docstring(
      argNames: Seq[String],
      argTypes: Seq[String],
      argDescs: Seq[String]): String = {

    val params =
      (argNames zip argTypes zip argDescs).map { case ((argName, argType), argDesc) =>
        val desc = if (argDesc.isEmpty) "" else s"\n$argDesc"
        s"$argName : $argType$desc"
      }
    s"Parameters\n----------\n${params.mkString("\n")}\n"
  }
}

class MXNetError(val err: String) extends Exception(err)
