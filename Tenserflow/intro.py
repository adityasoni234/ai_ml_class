import tensorflow as tf

single_value = tf.constant(4)
vector = tf.constant([1,2,3,4])
matrix = tf.constant([[1,2],[3,4],[5,6]])
matrix_3d = tf.constant([[[1,2],[3,4],[5,6]],[[7,8],[9,8],[11,12]]])

print(single_value)
print(vector)
print(matrix)
print(matrix_3d)

x = tf.constant([1,2,3,4])
y = tf.constant([5,6,7,8])

add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)
div = tf.divide(x,y)

print(add)
print(sub)
print(mul)
print(div)

x2d = tf.constant([[1,2],[3,4]])
y3d = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])

addtion = tf.add(x2d,y3d)
print(addtion)



zeros = tf.zeros([2,3])
print(zeros)

ones = tf.ones([3,4])
print(ones)

random_tensor = tf.random.normal(shape=(3,3),mean=0,stddev=1)
print(random_tensor)