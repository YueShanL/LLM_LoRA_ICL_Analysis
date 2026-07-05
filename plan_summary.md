# LoRA 指令计算重构研究计划摘要

本文档由 `plan.docx` 整理而来，面向后续系统搭建与实验实现。重点不是复述研究计划全文，而是提炼系统需要支持的实验对象、数据、模型配置、采集信号、指标与模块边界。

## 1. 研究目标

研究主题是：LoRA adapter 是否会在大语言模型中重构由自然语言指令诱导的 Transformer 内部计算。

核心假设是：如果 LoRA 学到的参数更新与指令 token 在模型内部触发了相似的隐藏状态扰动、注意力模式和因果计算路径，那么 LoRA 可以被理解为一种“参数固定的指令控制形式”。反之，如果两者输出行为相似但内部路径不同，则说明同一任务可能存在多条可替代的计算路径。

系统需要支持对三类模型条件进行对齐比较：

- Base：原始基础模型，不提供任务指令，不加载 LoRA。
- Instruction-only：基础模型加自然语言任务指令，不加载 LoRA。
- LoRA-only：基础模型加载 LoRA adapter，不提供任务指令。

可选扩展条件：

- LoRA + instruction：作为上界或完整微调条件。
- Base patching：用于验证 patching 操作本身不会恢复任务性能。

## 2. 研究问题

### RQ1：LoRA 是否产生与语义指令 token 相似的隐藏状态扰动？

比较 LoRA-only、instruction-only 与 base 模型在输出 token 位置的 residual stream hidden states。关注的是相对 base 模型的扰动是否相似，而不是只比较最终输出。

系统需要采集每层 Transformer block 之间的 residual stream activation，并在 teacher forcing 下对同一输入和同一目标输出位置进行对齐。

主要指标：

- Cosine similarity：比较扰动向量方向相似度。
- CKA：比较层级表示结构相似度。
- Logit distribution similarity：作为输出层面的辅助观察。

### RQ2：LoRA 是否在 attention head 上形成与语义指令相似的激活模式？

比较 LoRA-only 与 instruction-only 在生成目标输出 token 时的 attention head 行为。该问题用于判断两者是否通过相似的注意力路径提取和组合信息。

系统需要按 layer/head/token 维度采集 attention module 的输出或 pre-head attention activations。

主要指标：

- 每个 attention head 的 cosine similarity。
- Attention entropy：衡量注意力分布集中程度。
- KL divergence：捕捉余部差异和细微行为差异。
- Head ablation impact：逐个移除 head，找出对任务最关键的 head，并检查这些 head 在 LoRA 和 instruction 条件下是否具有相似因果作用。

### RQ3：LoRA 与语义指令是否使用相同或可互换的计算路径？

通过 activation patching 做因果检验：从 source 条件中抽取中间激活，注入 target 条件对应层和输出位置，观察任务性能是否保持。

典型 patching 方向：

- Instruction-only -> LoRA-only。
- LoRA-only -> Instruction-only。
- Base -> adapted condition，作为负控。

评估指标：

- Cross-entropy loss。
- Final accuracy。
- Confusion matrix。
- 与未 patch、同条件 patch、base patch 的性能差异。

### 额外可做方向（后续建议）

以下方向不是原始 research question 的必要组成部分，而是基于当前实现和初步结果，建议额外补充的验证实验，用于增强结论可靠性或排除替代解释。

- Instruction-only 可行性筛选：在每个 task 上先做多 prompt、few-shot 或 template sweep，确认自然语言指令条件本身能够稳定完成任务。若 instruction-only 行为过弱，RQ3 的路径可互换性结论会变得不干净。
- Matched-behavior RQ3：只在 instruction-only 与 LoRA-only 都达到任务阈值的设置上做 activation patching，避免把行为能力差异误判为计算路径差异。
- RQ3 最小控制矩阵：对同一 layer/site 跑 unpatched、same-condition patch、base -> target、instruction-only -> lora-only、lora-only -> instruction-only，验证 patching 操作本身不是主要破坏因素。
- Value/readout 假设验证：在 RQ2.1 发现 attention probability 相似但 attention output 差异更大后，进一步比较或 patch `v_proj` output、`o_proj` output 和 MLP output，检查 LoRA 的主要作用是否集中在 value/readout 或后续写回路径。
- 小规模 rank sweep：优先比较 r=1、r=4、r=8、r=16，观察 residual similarity、attention-output similarity 和 RQ3 patch effect 是否随 rank 改变，而不是一开始扩大到完整网格。
- 置信区间与重复性检查：对 RQ1/RQ2/RQ2.1 的 layer-level mean 做 bootstrap confidence interval，并对关键结论使用多个 random seed 复跑，减少单次采样或单 seed 造成的偶然性。

## 3. 实验设置

### 模型

首选模型：

- LLaMA 2 7B。

硬件受限时的备选模型：

- LLaMA 3.2 3B。
- GPT-2。

实现框架：

- Hugging Face Transformers。
- PEFT。
- PyTorch forward hooks。
- TransformerLens。

LoRA 设置：

- 主要注入 attention projection matrices：q_proj、k_proj、v_proj。
- 比较多个 rank：r = 1、4、16、64。
- 必要时使用 QLoRA 降低显存需求。

### 数据

数据集应为合成 transformation task，而不是普通领域知识微调数据。原因是普通领域任务可能与预训练知识重叠，导致 LoRA 只是放大已有表示，而不是学习明确的新任务规则。

每个数据集只包含一种固定 transformation，避免模型在无标签条件下混淆多种操作。

原计划中的示例：

| input | semantic instruction | output |
| --- | --- | --- |
| dataset will be designed | Add ZXQ after each word end with t or l | datasetZXQ willZXQ be designed |
| X29 | Test the length of input, if odd, wrap with @@ | @@ X29 @@ |
| Hello the run | Reply "JolNoC" if there exist incorrect gramma | JolNoC |

系统实现时建议将每个 transformation 作为独立 task registry 条目，包括：

- task_id。
- natural_language_instruction。
- input_generator。
- deterministic_label_function。
- train/validation/test split。
- allowed_output_format。

## 4. 实验运行约束

为了保证 LoRA-only 与 instruction-only 的内部状态可比较，系统应固定以下条件：

- 使用相同 base model。
- 使用相同 tokenizer。
- 使用相同输入样本。
- 使用 teacher forcing，让不同条件对齐到同一正确输出序列。
- 只在输出 token 位置统计核心指标，排除输入 token。
- 保持 generation/evaluation batch、padding、position alignment 一致。

关键变量必须显式记录：

- model_name。
- adapter_path 或 adapter_id。
- LoRA rank。
- task_id。
- prompt template。
- instruction 是否启用。
- hook points。
- random seed。
- dataset split。

## 5. 系统模块建议

### 5.1 Dataset 模块

职责：

- 生成 synthetic transformation 数据。
- 保存 instruction、input、target output。
- 支持每个任务只绑定一种 transformation。
- 输出统一样本格式，供训练、评估和 activation 采集复用。

建议输出字段：

- `sample_id`
- `task_id`
- `input_text`
- `instruction_text`
- `target_text`
- `condition`

### 5.2 LoRA 训练模块

职责：

- 对每个 task_id 训练对应 LoRA adapter。
- 支持不同 rank 配置。
- 可切换 full precision LoRA 与 QLoRA。
- 保存训练日志、adapter 权重和配置。

建议最小产物：

- adapter weights。
- PEFT config。
- train/eval loss curve。
- task metadata snapshot。

### 5.3 Evaluation Runner

职责：

- 在 base、instruction-only、LoRA-only、LoRA+instruction 等条件下运行同一批样本。
- 使用 teacher forcing 计算 loss、accuracy 和 logits。
- 调用 hook collector 保存内部激活。

需要避免把 prompt 格式差异误判为机制差异。instruction-only 与 LoRA-only 的输入长度可能不同，因此输出 token 对齐逻辑必须单独处理。

### 5.4 Activation Collector

职责：

- 注册 PyTorch forward hooks 或 TransformerLens hook points。
- 采集 residual stream activation。
- 采集 attention head activation 或 attention pattern。
- 按 layer/head/token/sample/condition 保存张量。

建议保存为分块文件，避免一次性保存全量激活导致内存和磁盘压力过大。

### 5.5 Similarity Analyzer

职责：

- 计算 residual perturbation similarity。
- 计算 per-layer、per-token、per-task 的 cosine similarity 和 CKA。
- 汇总 LoRA rank 与 similarity 的关系。
- 输出表格和可视化数据。

关键比较对象：

- delta_instruction = hidden_instruction - hidden_base。
- delta_lora = hidden_lora - hidden_base。
- similarity(delta_instruction, delta_lora)。

### 5.6 Attention Analyzer

职责：

- 计算 per-layer/per-head attention similarity。
- 计算 attention entropy 与 KL divergence。
- 运行 head ablation，识别关键 head。
- 检查关键 head 在 LoRA-only 与 instruction-only 中是否一致。

### 5.7 Activation Patching 模块

职责：

- 支持 source/target 条件组合。
- 在指定 layer、token position、activation site 注入 source activation。
- 重新计算 loss、accuracy、confusion matrix。
- 与未 patch 和 base patch 对照比较。

最小 patching 单元：

- sample_id。
- layer。
- token position。
- source_condition。
- target_condition。
- activation_name。

### 5.8 Result Store 与报告模块

职责：

- 统一保存实验配置、指标、图表和摘要。
- 支持按 task、rank、layer、head、condition 查询。
- 生成最终论文/报告所需结果表。

建议目录结构：

```text
experiments/
  {run_id}/
    config.yaml
    dataset_snapshot/
    adapters/
    metrics/
    activations/
    plots/
    reports/
```

## 6. 主要对照组

必须对照：

- Base raw input：下界。
- Instruction-only：语义指令控制。
- LoRA-only：参数化任务控制。
- LoRA + instruction：适配质量上界。
- Base activation patch：patching 负控。

建议扩展：

- 不同 LoRA rank。
- 不同 transformation task 类型。
- 不同模型规模。
- 不同 hook site，例如 residual stream、attention output、MLP output。

## 7. 预期贡献

理论贡献：

- 区分 LoRA 是否复用 instruction-induced computation，还是通过不同内部机制达到相似输出。
- 为 PEFT、模型编辑和安全解释提供机制层面的证据。

方法贡献：

- 提供一个比较“指令控制”和“参数控制”的可复用实验框架。
- 将 residual perturbation、attention analysis 与 activation patching 组合成一套系统化评估流程。
- 可扩展到 Prefix Tuning、Adapters 等其他 PEFT 方法。

工程贡献：

- 建立可重复运行的 synthetic task benchmark。
- 建立可复用的 activation collection 和 patching pipeline。
- 输出按模型、任务、rank、layer/head 组织的可查询实验结果。

## 8. 系统搭建优先级

第一阶段：可运行最小闭环

1. 实现一个 synthetic task。
2. 用小模型跑通 base、instruction-only、LoRA-only。
3. 完成 LoRA 训练。
4. 在 teacher forcing 下采集 residual stream。
5. 计算 residual perturbation cosine similarity。

第二阶段：扩大比较范围

1. 增加多个 transformation task。
2. 增加多个 LoRA rank。
3. 加入 CKA 和 logits similarity。
4. 建立标准结果目录和配置记录。

第三阶段：attention 与因果分析

1. 加入 attention head activation 采集。
2. 实现 entropy、KL divergence 和 head ablation。
3. 实现 activation patching。
4. 加入 confusion matrix 和 cross-condition patching 报告。

第四阶段：报告与复现实验

1. 固化 run config。
2. 自动生成图表和表格。
3. 输出可复现实验报告。
4. 对模型规模、任务类型和 rank 做系统性对比。

## 9. 实现风险与注意事项

- Instruction-only 与 LoRA-only 的 token 序列长度不同，输出位置对齐必须谨慎处理。
- 如果任务太简单，base model 可能直接完成任务，削弱对照效果。
- 如果任务太难，LoRA 训练失败会导致内部相似性分析失去意义。
- Attention pattern similarity 不能单独证明计算等价，需要 activation patching 做因果检验。
- 高 hidden-state similarity 不必然代表可互换计算路径。
- 激活张量体积大，需要从一开始设计缓存、分块和清理策略。
- 多 rank、多任务、多层 patching 会快速扩大实验组合，应优先实现配置化批处理。

## 10. 最小系统验收标准

一个可用于后续研究的系统至少应满足：

- 能生成并保存合成 transformation 数据。
- 能训练并加载每个 task 对应的 LoRA adapter。
- 能在相同样本上运行 base、instruction-only 和 LoRA-only。
- 能在 teacher forcing 下对齐目标输出 token。
- 能采集指定层的 residual stream activation。
- 能计算 LoRA 与 instruction 相对 base 的扰动相似度。
- 能保存完整 run config、指标和中间结果。
- 能复现实验结果，而不是只输出一次性 notebook 结果。
