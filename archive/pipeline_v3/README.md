# pipeline_v3 — 统一桥约束版(2026-07-15 设计,理论先行)

新版 pipeline 的独立目录。根目录的共享基础模块(graph / testbeds / pool / encode / judge /
metrics / splice_decode / negop)继续复用;`experiments/` 是历史实验脚本归档;旧 runner
(run_task1/run_task2)保留用于复现已发表数字。

## 方法(与旧版的差异)

**桥公理(显式化)**:变量的语义相似度 cos(e_i, e_j) 是其统计依赖强度 dep(X_i, X_j) 的单调函数;
条件版对应条件依赖。旧版的独立对去相关(下尾)与残差 Pearson 锚定(条件版)都是它的特例;
依赖下界(上尾)曾死于向量取负的坏基底,在 f_neg 基底上按同一公理整体重审。

组件:

1. `dependence.py` — 依赖矩阵基建:Pearson / distance correlation / kNN-MI,各两层
   (边际;条件在父得分上=先残差化再度量)。按数据集缓存 npz。MI/CMI 由此落地(Yujia 的
   "第二约束"方向)。
2. `bridge.py` — 统一桥约束:图分层(无 trek 对→边际层;兄弟对→条件层),
   损失 = 下尾(独立对 cos²)+ 上尾(强依赖对 hinge(κ·dep_q − |cos|))+ 条件锚定
   (残差相似度 ≈ 条件依赖)。依赖度量三选一(pearson/dcor/mi)作为对照 arm。
3. `solve.py` — 目标 = 生成一致(负边走 f_neg)+ 残差 + 桥约束 + norm;ALS 闭式初始化 +
   Adam 精修(确定性)。
4. `translate.py` — Task 2 双读出并列:优化 u 解码;forward-β 读出(路径权聚合,反向路径
   f_neg)。主读出按三方对照的证据定。
5. `intervene.py` — 完整 swap 干预(judge 版 + 正极参考系),第三正式指标。
6. `run_pipeline.py` — 统一入口:Task 1 + Task 2 + swap,records 落盘;LLM-naming 基线
   fold 对齐(只看可见子节点)。

## 纪律(不变)

mask-20% 是任务本身;dev 16 调一切,held-out 3(hexaco/riasec/kims)只考一次;每个新约束
arm 必须先有机制假设 + 预登记成功线,同批对照,输者当日处决;全局组件(encoder / 词典 /
f_neg)只用 WordNet + dev 训练。

## 预登记(阶段 2 桥约束)

- 过线:dev 池 Task 1 judge 或 match 相对当前冻结配置(生成+f_neg+残差 Pearson 锚定)
  平均 +0.03 以上;
- 护栏:himi / gcbs 回退 ≤ 0.05;
- 对照 arm:pearson / dcor / mi 三版桥约束 + 现冻结配置基准,同批 judge;
- 失败处置:整包不进冻结,只保留 dependence.py 基建与诊断结论。
