# GenRec 多目标展开与样本加权

GenRec 支持通过现有 `data_config.sample_weight_fields` 指定每条样本的 loss 权重。
使用第一个配置的权重字段，与单任务模型的样本权重约定一致；字段名不必固定为
`sample_weight`。

```protobuf
data_config {
  sample_weight_fields: "sample_weight"
}
```

数据中的权重列应为每行一个浮点数。不需要把该列配置为 feature，也不要将其加入
prompt；权重仅用于计算 loss，不参与输入 embedding。

## 多 label 样本的准备

1. 按 ItemID 去除同一原始样本 label 内的重复目标。
1. 将每个有效目标展开为一条训练样本，复制该原始样本的历史和用户特征。
1. 为每条子样本填写预先计算好的权重，再打散子样本。

设有效原始样本数为 N，展开后样本数为 M，某条原始样本包含 m 个有效唯一目标：

```text
sample_weight = (M / N) / m
row_ce = mean(该子样本有效 response token 的交叉熵)
batch_loss = mean(sample_weight * row_ce)
```

`1/m` 使同一原始样本的目标平均分配其权重；全局常数 `M/N` 使整个展开数据集的
平均权重为 1。完整数据集的平均 loss 等于先平均每个原始样本的多目标 loss，再对
原始样本求平均。子样本无需落在同一 batch 中。

模型不执行 label 展开、去重、打散或权重归一化；这些步骤在数据准备阶段完成。
模型也不按当前 batch 的权重和重新归一化，不增加跨卡通信。

## loss 与评估行为

- 配置权重时，先进行 causal label shift，忽略 `ignore_index`，在每行内部按有效
  response token 数求平均，然后乘权重并对 batch 求平均。
- 未配置权重时，仍调用原有 Hugging Face causal-LM loss，保持原始 token 平均口径。
- packed FlashAttention 按样本边界计算 attention，输出恢复为逐行 response 窗口后
  计算加权 loss；不会跨样本预测下一个样本的 token。
- beam 搜索、SID 解码及 ItemID HR/Recall 定义不因 loss 加权而改变。评估多目标召回
  时仍应使用保留完整 label 集合的原始评估样本。

## 续训学习率

恢复训练及学习率进度时同时传入 `--continue_train --restore_lr_scheduler`。
旧 checkpoint 没有 scheduler 状态时，按照原学习率配置和保存的 step/epoch 进度
重建；应保持原 schedule 的总步数，而不是改成剩余步数。
