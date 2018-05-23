# tensorboard使用练习 
通过构建一个全连接层神经网络，用tensorboard进行相关参数的可视化显示。
练习总结：  
1、程序运行后，创建过summary的变量，在不同name_scope中都会有相应的图表，只是所属的名称域不同。  
2、summary的添加位置，对分布和直方图的图像没有影响。分别尝试了在主函数中添加和各name_scope中添加。 
原因：在打开session运行各操作之前，我们所做的只是在构建整个图的结构，所有的操作都没有运行，包括进行梯度下降更新参数，所以定义summary的位置只要在变量或操作定义的后面就可以，不会影响在tensorflow中的可视化显示。  
3、tensorboard中Graph的显示，与程序中的name_scope( )一一对应，每一个name_scope为一个显示模块。





