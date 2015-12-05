package ml.dmlc.mxnet.jnitest

object ScalaNativeStuff {
  def main(args: Array[String]) {
    System.loadLibrary("mxnet-scala")
    System.out.println(new ScalaNativeStuff().helloNative)
    val inst = new ScalaNativeStuff
    val p: Long = inst.allocBuf
    System.out.println(p)
    inst.printBuf(p)
  }
}

class ScalaNativeStuff {
  @native def helloNative: Long
  @native def allocBuf: Long
  @native def printBuf(pointer: Long)
}
