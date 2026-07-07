# TensorFlow CNN 训练说明

本项目的 TensorFlow 版本脚本：

- `train_cnn_tensorflow.py`: 训练 CNN，按每类 400 张训练、100 张验证切分
- `evaluate_cnn_tensorflow.py`: 加载已保存模型，验证或预测单张图片

在项目统一环境 `dl` 中运行：

```bash
conda activate dl
cd /Users/yuanjunhao/Desktop/project_data
python train_cnn_tensorflow.py
python evaluate_cnn_tensorflow.py
```

训练完成后会生成：

- `tf_digit_cnn.keras`: 最终模型
- `tf_digit_cnn_best.keras`: 验证集准确率最高的模型
- `tf_training_log.csv`: 每轮训练日志
- `tf_training_report.json`: 训练配置和结果摘要
- `tf_confusion_matrix.npy`: 验证集混淆矩阵

预测单张图片：

```bash
conda activate dl
cd /Users/yuanjunhao/Desktop/project_data
python evaluate_cnn_tensorflow.py path/to/image.png
```
