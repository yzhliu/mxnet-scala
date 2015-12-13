package ml.dmlc.mxnet

import org.scalatest.{FunSuite, BeforeAndAfterAll}

class NDArraySuite extends FunSuite with BeforeAndAfterAll {
  test("to java array") {
    val ndarray = NDArray.zeros(Array(2, 2))
    assert(ndarray.toArray === Array(0f, 0f, 0f, 0f))
  }
}
