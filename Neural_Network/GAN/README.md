以Goodfellow原论文为参考，构建GAN网络。其中生成器(Generator)和鉴别器(Discriminator)均为三层全连接神经网络。GAN.ipynb和GAN.py两个文件均为对GAN的实现，为了比较一些超参数和不同神经网络的效果对最后实验结果的影响，GAN.ipynb的G和D分别采用了三层全连接神经网络，GAN.py的G和D分别采用了四层全连接层的神经网络，在各隐藏层的节点设置上，也与GAN.ipynb略有不同，同时GAN.py存储了每一次迭代的生成器生成的照片（程序中的设置为500幅），并在图像显示函数中，对图像进行了一些处理。

总结及相关说明：  
1、使用matplot.pyplot进行图像显示时，在.py文件和.ipynb（即ipython notebook）中的实现方式是不同的。notebook中需要使用canvas进行实现。下面举个简单的例子：  
在.py文件中实现：   

```
f, a = plt.subplots(4, 10, figsize=(10, 4))
for i in range(10):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[4, noise_dim])
    g = sess.run([gen_sample], feed_dict={gen_input: z})
    g = np.reshape(g, newshape=(4, 28, 28, 1))
    # Reverse colours for better display
    g = -1 * (g - 1)
    for j in range(4):
        # Generate image from noise. Extend to 3 channels for matplot figure.
        img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                         newshape=(28, 28, 3))
        a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()
```
在.ipynb中实现：  

```
# Generate images from noise, using the generator network.
n = 10
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[n, noise_input])
    # Generate image from noise.
    g = sess.run(g_output, feed_dict={g_input: z})
    # Reverse colours for better display
    g = -1 * (g - 1)
    for j in range(n):
        # Draw the generated digits
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()

```
2、output文件中存放的是GAN.py每一次迭代生成的图像（共500幅），可以看出随着迭代次数的增加，图像的质量越来越好。

    
