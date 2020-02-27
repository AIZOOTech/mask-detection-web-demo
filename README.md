# face-mask-detection-web-demo
# 人脸口罩检测网页demo
Face mask deteciton demo, by tensorflow.js

## 原理
本demo是在浏览器运行的人脸口罩检测网页demo，介绍如何将深度学习的人脸口罩检测模型部署到浏览器里面。
关于人脸口罩检测的PyTorch、TensorFlow、Caffe、Keras、MXNet版本，可以进入相应Github仓库
[FaceMaskDetection](https://github.com/AIZOOTech/FaceMaskDetection)
关于项目介绍，可以阅读一下两篇文章：
[AIZOO开源人脸口罩检测数据+模型+代码+在线网页体验，通通都开源了](https://mp.weixin.qq.com/s/22U_v6IQ9PBHslI-65v_0Q)
[人脸口罩检测现开源PyTorch、TensorFlow、MXNet等全部五大主流深度学习框架模型和代码
](https://mp.weixin.qq.com/s?__biz=MzIyMDY2MTUyNg==&mid=2247483779&idx=1&sn=b9ac5af31adf1dfdc3c87eb1c74836a5&exportkey=AX%2FANiIY8CWWMPQrHKh6A5E%3D&pass_ticket=aaNfWJGBgSum6CY5pvFqx0IIfljPPkeX%2BdMtPEl3zo5hQfPnYR5mEUlayz06kNKG)


深度学习模型，可以借助TensorFlow.js库，运行在浏览器里。首先，需要使用`tensorflowjs_converter`将tensorflow的`graph model`或者keras的`layer model`转换为TensorFlow.js支持的模型。
该工具可以通过`pip install tensorflowjs`安装。

如果使用Keras模型转的操作如下：
```
tensorflowjs_convert --input_format keras --output_format tfjs_layers_model   /path/to/keras/hdf5/model  /path/to/output/folder
```
模型会生成`model.json`文件和一个或者多个`bin`文件，其中前者保存模型的拓扑，后者保存模型的权重。
使用JavaScript，需要在html中先引入`tfjs.min.js`库，然后加载模型
```
<script src="js/tfjs.min.js"></script> 

```
在`detection.js`中，加载模型
```
model = await tf.loadLayersModel('./tfjs-models/model.json');
```
置于anchor生成、输出解码、nms与使用python版本并无太大差异，大家可以查看`detection.js`中三个相关的函数，一目了然。

## 运行方法
在当前目录下打开终端，只需要建立一个最小web server即可。
对于使用python的用户
```
// python3用户
python -m http.server
// python2用户
python -m SimpleHTTPServer

```
如果你使用Node.js
```
npm install serve -g //安装serve
serve // this will open a mini web serve
// 您也可以使用http-serve
npm install http-server -g
http-server
```

## 效果
您可以点击网页中的上传图片按钮，或者拖拽图片到网页区域，然后模型会自动进行检测并画框。
![页面效果图](/images/result.png)