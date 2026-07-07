# 手写数字识别 GUI

这个项目提供一个轻量的手写数字识别 GUI，可以选择单张图片或文件夹进行预测。

## Windows 使用

```bash
pip install -r requirements.txt
python digit_gui.py
```

打包为 Windows exe：

```bash
python build_exe.py
```

生成文件在 `dist/DigitRecognizer.exe`。

## 文件夹准确率

如果选择的文件夹包含 `0` 到 `9` 子目录，程序会把子目录名作为真实标签并计算准确率。否则只输出预测数字和置信度。

## 说明

仓库只保留代码和轻量 TFLite 推理模型，不包含训练数据集、原始图片和增强数据。
