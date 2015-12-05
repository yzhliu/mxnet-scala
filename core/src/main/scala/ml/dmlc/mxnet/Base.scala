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
  type mx_uint = Int
  class NDArrayHandle

  // TODO
  System.loadLibrary("mxnet-scala")
  val _LIB = new LibInfo


  // helper function definition
  def checkCall(ret: Int): Unit = {
    /**
      Check the return value of C API call

      This function will raise exception when error occurs.
      Wrap every API call with this function

      Parameters
      ----------
      ret : int
          return value from API calls
    */
    if (ret != 0) {
      throw new MXNetError(_LIB.mxGetLastError())
    }
  }
}

class MXNetError(val err: String) extends Exception(err)
