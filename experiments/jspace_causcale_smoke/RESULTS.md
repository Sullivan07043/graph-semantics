# J-space × CauScale × v4.1 小规模实验结果

日期：2026-07-21

## 结论

这轮实验发现了可重复的跨层结构信号，但**没有发现 v4.1 相对简单基线的增益**。

- 80% CauScale 共识图明显优于度匹配乱序图，说明图中不是纯噪声。
- 但 clean-only correlation 图在干预预测上与 CauScale 相当或更好。
- 在同一 CauScale 图上，冻结 v4.1 没有超过 unit-multiplier 版本。
- 无图 positive-correlation 基线的 match 也高于 v4.1。

因此当前判断是：**CauScale 找到了以“同概念逐层传播”为主的候选结构；尚不能说明 v4.1 在该 J-space 设置中有额外价值。**

## 最终实验协议

- 模型：`Qwen/Qwen3.5-4B`
- Jacobian lens：官方 Qwen3.5-4B、n=1000 lens
- 坐标：5 个预注册 token 概念 × 4 层（8、16、24、30）= 20 节点
- discovery：500 个独立 prompt，每个有一条 clean 和一条 hard-do，合计 1000 行
- heldout：100 个与 discovery 不重叠的 prompt，合计 200 行
- setpoint：每个节点内部近似均衡 `-2,-1,+1,+2` clean SD；全局各 125 次
- hard-do：在每层用 dual basis 将一个选定坐标设为固定 setpoint；同层其余四个选定坐标保持不变
- CauScale：官方 synthetic checkpoint；20 个 feature-bootstrap seeds（42--61）
- v4.1：正式冻结 release；边权和所有相关基线只使用 clean 行

旧的 additive pilot 因为同层方向串扰、符号与剂量混杂而被排除，只保留作工程日志。

## 干预 QA

500 条 discovery hard-do 全部通过：

- failed rows：0/500
- 最大目标坐标误差：0.01362 clean SD
- 最大同层非目标漂移：0.00706 clean SD
- 预注册容忍阈值：两者均为 0.10 clean SD
- 四种 setpoint 全局计数：各 125；每节点每条件 6--7 次

这项 QA 只保证“所选的五个同层线性坐标”中一次只改变一个；未测量的 residual-stream 方向仍可能变化，因此不能声称满足完整 causal sufficiency。

## CauScale 图稳定性

20 次 feature bootstrap：

| 指标 | 结果 |
|---|---:|
| 每次保留边数 | 35--48（均值 40.85） |
| 两两 edge Jaccard | 0.431--0.800（均值 0.605） |
| 至少 16/20 次出现的同向边 | 26 |
| 20/20 次出现的同向边 | 17 |

26 条 80% 共识边中：

- 21 条顺层边，0 条逆层边；
- 14 条是同一概念在相邻层之间的顺向边；
- 7 条是跨概念顺层边；
- 5 条是同层边。hard-do 在同层的非目标坐标保持不变，因此这 5 条是明确的结构负对照失败。

## 配对干预验证

发现集内（用于诊断，不能视为独立确认）：

| 图 | Direct ROC-AUC / AP | Reachability ROC-AUC / AP |
|---|---:|---:|
| CauScale seed42 | 0.920 / 0.651 | 0.965 / 0.670 |
| CauScale 80% consensus | 0.928 / 0.633 | 0.936 / 0.676 |
| clean correlation（seed42 同边数） | 0.975 / 0.697 | 0.977 / 0.703 |

独立 heldout 上，每节点只有 5 个 prompt pair，多重校正后没有效应达到 detectable 阈值，所以 ROC-AUC/AP 不可估计。连续效应量仍显示候选图比乱序图强：

| 图 | 预测 direct edge 的平均经验效应 | 度匹配乱序均值 | 乱序经验 p | clean correlation |
|---|---:|---:|---:|---:|
| seed42 | 0.225 | 0.0336 | 0.00498 | 0.244 |
| 80% consensus | 0.298 | 0.0349 | 0.00498 | 0.310 |

这支持“结构不是随机的”，但不支持“CauScale 优于相关性”。

## 冻结 v4.1 语义补全

所有值使用相同五折隐藏标签；权重估计只用 500 条 clean 行。

| 图/方法 | Match | Exact | True-target cosine |
|---|---:|---:|---:|
| seed42 CauScale + v4.1 | 0.80 | 0.05 | 0.94723 |
| seed42 CauScale + unit multipliers | 0.80 | 0.05 | 0.94781 |
| seed42 degree-matched shuffle + v4.1 | 0.65 | 0.15 | 0.93206 |
| 80% consensus + v4.1 | 0.90 | 0.05 | 0.95565 |
| 80% consensus + unit multipliers | 0.90 | 0.10 | 0.95684 |
| consensus-edge-count correlation + unit multipliers | 0.80 | 0.05 | 0.95765 |
| no-graph positive correlation | 1.00 | 0.00 | 0.95486 |

共识 topology 比单次乱序 topology 强，但 v4.1 learned multipliers 比同图 unit multipliers 低 0.00118 cosine；match 也没有提升。当前不能把 topology 的作用归因于 v4.1。

## 限制与下一步门槛

当前仍是 feasibility pilot：

1. 只有 5 个语义概念，在 4 层重复；这让 correlation baseline 很强。
2. heldout 每节点只有 5 个独立 prompt，无法给单边显著性提供足够功效。
3. 20 次只改变 CauScale precision-prior 的 feature sample，不是 20 套独立数据。
4. 这是 token-anchored J-lens 线性坐标，不是对完整 J-space 做穷尽性因果发现。
5. 尚未加入 matched-norm random direction、mask permutation 和三套独立 prompt seeds。

继续扩实验的最低门槛：

- heldout 至少 20 个独立 prompt/节点；
- 加 matched-norm random-direction control；
- 至少 3 套独立 prompt/randomization seeds；
- v4.1 必须同时超过 same-graph unit multipliers、clean correlation 和多数 graph shuffles。

若这四项仍不能满足，就应停止扩大 CauScale + v4.1 路线，而不是继续调参寻找正结果。

## 主要产物

- `runs/jspace_discovery_harddo_v2/manifest.json`
- `runs/jspace_discovery_harddo_v2/causcale_consensus80.npz`
- `runs/jspace_discovery_harddo_v2/intervention_validation_consensus80.json`
- `runs/jspace_discovery_harddo_v2/v41_consensus80_clean_only.json`
- `runs/jspace_heldout_harddo_v2/intervention_validation_consensus80_external.json`
