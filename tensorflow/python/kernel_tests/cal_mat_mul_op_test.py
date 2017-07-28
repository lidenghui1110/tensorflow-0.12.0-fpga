import tensorflow as tf

class CalMatMulTest(tf.test.TestCase):
  def testCalMatMul(self):
    cal_module = tf.load_op_library('/home/lidenghui/newdisk/user_ops/calmatmul.so')
    with self.test_session():
      result = cal_module.cal_mat_mul_op([[1.0,2.0, 3.0],[4.0,5.0,6.0]],[[1.0,2.0],[3.0,4.0],[5.0,6.0]])
      self.assertAllEqual(result.eval(), [[22.0,28.0],[49.0,64.0]])

if __name__ == "__main__":
  tf.test.main()
