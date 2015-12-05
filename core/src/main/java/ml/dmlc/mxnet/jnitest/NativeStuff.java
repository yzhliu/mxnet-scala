package ml.dmlc.mxnet.jnitest;

public class NativeStuff {
  public native long helloNative();
  public native long allocBuf();
  public native void printBuf(long pointer);

  public static void main(String[] args) {
    System.loadLibrary("mxnet-scala");
    System.out.println(new NativeStuff().helloNative());
    NativeStuff inst = new NativeStuff();
    long p = inst.allocBuf();
    System.out.println(p);
    inst.printBuf(p);
  }
}
