## 0. 术语与符号（必须统一）

* **问题样本**：一个编程题 + 测试集
* **测试集**：(\mathcal{D}={(x_j, y_j)}_{j=1}^N)
* **审计子集（audit）**：从 (\mathcal{D}) 固定抽取 (M_{audit}) 个样例，用于推理一致性与 soft reward
* **轮次（round）**：同一问题内的多轮修复/生成，最多 (T_{max}) 轮
* **兄弟组（sibling group）**：同一父节点一次扩展生成的子节点组；代码阶段大小为 `K`（默认 2），最终单轮推理阶段大小为 `K_reason`（默认 4）
* **节点预算**：同一问题允许生成的**代码搜索节点**总数上限 (N_{max})
* **节点计数口径**：代码搜索阶段每生成一个“候选解”（包含 code/reason 两段之一或两段）计 1；被剪枝/重采样的也计入。最终单轮推理阶段生成的节点单独统计，不计入 `node_count`
* **模式**：`mode=train|test`（两者都允许多轮；test 不更新参数）

---

## 0.1 一眼看懂流程（建议先读）

1. 每个父节点每轮扩展 `K` 个子节点。
2. 每个子节点都计算两类奖励：`R_code` 与 `R_reason`。
3. **节点级剪枝**：若某子节点满足 `R_code==0 and R_reason==0`，该节点不进入下一轮。
4. **组级重采样**：若同一父节点扩展出的整组 sibling 全部“双0”，则整组丢弃并重采样，最多重试 `N` 次（实现参数名 `M_retry`）。若达到重试上限仍为“双0”，实现会保留最后一次生成结果（随后仍会被节点级双0剪枝）。
5. **代码通过后停止搜索**：一旦当轮出现 `pass_rate==1` 的节点，当前轮其余未通过节点不再继续扩展。
6. 若 `frozen_reason_one_shot=True`，则所有“通过代码但推理未冻结”的节点都会各自进入一次最终单轮推理 GRPO，每个节点生成 `K_reason` 个推理候选，然后整题停止。
7. 这次最终单轮推理不受 `N_max` 与 `T_max` 截断，但其生成节点会单独记录为 `final_reason_node_count`。
8. 训练时对保留样本计算正交 loss；测试时不更新参数，只统计指标和输出 trace。

---

## 1. 数据集适配约束（必须）

### 1.1 统一数据结构

任何数据集必须通过 `DatasetAdapter` 转换为统一结构（Trainer 只吃统一结构）：

```python
{
  "question_id": str,
  "prompt": str,
  "test_cases": [
    {"input": Any, "output": Any},
    ...
  ]
}
```

### 1.2 扩展新数据集的唯一入口

* 只能新增 `adapters/<dataset_name>_adapter.py`（或同等目录）
* 禁止在训练主循环里写 `if dataset==...` 特判
* 新数据集的最小改动：只实现 adapter +（必要时）少量配置项

---

## 2. Backend 切换约束（HF / vLLM 必须自由切换）

必须通过统一接口封装，训练主循环不得出现 `if use_vllm:` 分支。

### 2.1 必须提供的统一接口

```python
class Backend:
    def generate(self, prompt, **gen_cfg) -> str: ...
    def logprob(self, prompt, target_text, **cfg) -> float: ...
```

* `generate()`：用于生成 code/reason
* `logprob()`：可用于概率型 soft reward 或辅助打分（返回**平均 token logprob**或可换算量）

---

## 3. 上下文与反馈约束（多轮必须靠“错误反馈”推进）

### 3.1 上下文窗口

必须支持超参：

* `context_round_window = W`：下一轮生成时最多携带最近 W 轮历史
* 当前实现的主生成上下文采用“父代码全文 + 历史压缩反馈”：
  * 始终保留**当前父节点完整代码**
  * 最近一轮保留详细反馈：`status / failed_input / failed_actual / error_summary / mismatch_count`
  * 更早历史压成滚动摘要：状态统计、是否反复编译失败、是否反复命中同类失败输入、逻辑/执行不匹配趋势
* 历史内容不直接泄露标准答案；主生成阶段只反馈失败输入和实际结果/错误类型，不给 expected output
* 主生成 prompt 需显式提供“父节点代码”（parent code），指导模型基于旧代码修复

### 3.2 错误反馈截断/提取

* 编译/运行/超时错误可能很长
* 必须实现 `summarize_error(error_text, max_chars, max_lines)`：

  * 截断到 `max_chars` 或 `max_lines`
  * 或提取关键信息（异常类型、关键堆栈行、超时标记）
* 推荐把“首个失败样例”结构化记录到 history：
  * `failed_input`
  * `failed_actual`（实际值或错误类型）
* history 记录建议至少包含：
  * `round`
  * `status_code`
  * `error_summary`
  * `code_preview`
  * `failed_input / failed_actual`

---

## 4. 节点与状态定义（必须）

### 4.1 Node 必须包含字段

```python
Node:
  node_id: str/int
  parent_id: str/int | None
  round_idx: int

  code: str | None
  reasoning: str | None

  pass_rate: float
  R_soft: float
  R_code: float
  R_reason: float

  status_code: {"CORRECT","FAIL","SYNTAX_ERROR","RUNTIME_ERROR","TIMEOUT"}
  status_reason: {"HONEST","HALLUCINATION","NOT_RUN"}

  frozen_code: bool
  frozen_reason: bool

  # 可选：保存用于调试的执行摘要
  exec_summary: dict
```

### 4.2 Node_Count 计数口径（必须严格）

* 每生成一个子节点：`Node_Count += 1`
* 重采样产生的节点也计入
* 被剪枝节点也计入
* 节点内部包含 code+reason（或其中之一）仍算 1 个节点

---

## 5. 训练/测试共同的多轮流程（核心）

两种模式都跑同一 pipeline；区别只有：

* `train`: 计算 loss 并反向传播与 `optimizer.step()`
* `test`: **不进行任何参数更新**，但完整跑通多轮、错误反馈、上下文窗口、指标统计

实现保护（与代码一致）：若某个生成批次因全剪枝导致没有可训练样本，Trainer 会自动注入一个全零 fallback 样本，避免训练流程中断。

---

## 6. 审计样本（audit）选择规则（必须）

对每个问题样本：

1. 每次“再次遇到该问题”（一次新的 visit，例如新 epoch / 新 run）时，都从 `test_cases` 重新随机选 `M_audit` 个样例
2. **同一次 visit 内固定**：该问题的整个多轮/整棵树过程中不再变化
3. 同一父节点扩展出来的 K=2 个兄弟节点 **必须共享相同 audit 集**

> 这样可以降低 audit 子集过拟合，同时保证同一次树搜索内奖励可比性。

---

## 7. 生成格式与 token 掩码（必须）

### 7.1 输出必须结构化

主生成阶段（生成候选代码）输出必须可可靠解析为：

```text
<CODE>
...
</CODE>
```

约束（与实现一致）：

* 主生成阶段只要求输出**单个 fenced Python code block**，不再要求 `<REASON>`。
* 主生成阶段**不要求**输出 `<LOGIC_PREDICTION>` / `<EXEC_PREDICTION>`。
* 主生成的合法格式是：
  * opening fence: ` ```python `
  * closing fence: ` ``` `
  * 标签外不允许额外文本（由 `generation_outside_noise_chars` 控制，默认 0）
* 主生成解析使用单独的严格阈值 `generation_outside_noise_chars`；逻辑/执行审计仍可沿用单独的 `format_outside_noise_chars`。
* 逻辑推理与执行推理在后续审计阶段用**单独 prompt**完成：
  * 逻辑审计：`<REASON> + <LOGIC_PREDICTION>`
  * 执行审计：`<REASON> + <EXEC_PREDICTION>`
* 主生成 / 逻辑审计 / 执行审计都带有明确 few-shot 结构示例；逻辑与执行审计不再只靠自然语言规则约束格式。
* 三类生成（主生成 / 逻辑审计 / 执行审计）默认都先经过 tokenizer chat template 渲染（可由 `use_chat_template_for_codegrpo` 关闭）。

### 7.2 token mask 必须可得

实现必须能得到：

* `code_token_mask`
* `reason_token_mask`

当前实现的训练分配（与代码一致）：

* 主生成样本（单个 fenced Python code block）：`code_token_mask=全1`，`reason_token_mask=全0`，即 `A_code` 作用于主生成输出的全部 token
* 执行审计样本（`<REASON> + <EXEC_PREDICTION>`）：`code_token_mask=全0`，`reason_token_mask=全1`，该样本的 case-level `advantage` 写入 `A_reason`
* 逻辑审计样本（`<REASON> + <LOGIC_PREDICTION>`）：`code_token_mask=全0`，`reason_token_mask=全1`，该样本的 case-level `advantage` 写入 `A_reason`

---

## 8. 代码执行与输出匹配（必须“多样输出”鲁棒）

### 8.1 执行器统一返回结构

必须实现 `execute(code, input) -> ExecResult`：

```python
ExecResult:
  kind: {"OK","SYNTAX_ERROR","RUNTIME_ERROR","TIMEOUT"}
  value: Any | None        # kind == OK 时
  error_type: str | None   # kind != OK 时，例如 "SyntaxError"/"IndexError"/"Timeout"
  error_msg: str | None    # 截断后的摘要
```

### 8.2 匹配函数（必须鲁棒）

必须实现 `is_match(pred, actual) -> bool`，规则：

1. 若 `actual.kind != OK`：

   * 只要 `pred` 预测到**错误类型一致**即可算对（不要求数值）
2. 若 `actual.kind == OK`：

   * 结构化解析（如 `ast.literal_eval`）优先
   * list/tuple 等价
   * dict 结构等价
   * float 允许 `1e-6` 容差
   * 字符串 `strip` + 归一化空白
3. 无法解析时：允许归一化字符串相等（不使用 Levenshtein 作为主判断）

> 重点：**提醒 AI 实现时优先结构化对比，最后才字符串兜底。**

---

## 9. 代码奖励 R_code（必须）

### 9.1 pass_rate

对完整测试集运行：

```python
pass_rate = num_pass / N
```

### 9.2 soft reward 触发条件

* 默认在每个节点都计算（只要有 audit 样本）。
* 只用 audit 集计算 soft reward（不跑全量）。
* `lambda_soft` 必须满足 `0 <= lambda_soft <= 1`。
* `soft_reward_ineligible_scale` 必须满足 `0 <= soft_reward_ineligible_scale <= 1`。
* `R_soft` 会先按逻辑审计得到原始值；真正进入 `R_code` 的是 `R_soft_effective`（见 9.4）。

### 9.3 soft reward 的定义（逻辑推理分支）

对 audit 集每个样例 ((x_j, y_j))：

1. 模型先做“逻辑推理分支”，给出 `<LOGIC_PREDICTION>`，其含义是：
   基于**题目 + 代码意图 + 输入**，预测理想输出，忽略语法错误等实现层面的细小问题。
2. 将 `<LOGIC_PREDICTION>` 与标准答案 `y_j` 比较（用 `is_match` 或等价鲁棒比较函数）。

```python
logic_correct_j = 1 if is_match(logic_prediction_j, y_j) else 0
R_soft = mean(logic_correct_j over audit)
```

实现细节（与代码一致）：

* 若逻辑分支格式不合法（如缺失标签，或 `<REASON>` 未先于 `<LOGIC_PREDICTION>`），先应用格式惩罚：

```python
if logic_format_ok:
    logic_score = clamp01(logic_correct + format_bonus_logic)
else:
    logic_score = clamp01(logic_correct - format_penalty_logic)
```

* 逻辑格式项是**独立加法项**：格式正确时直接加 `format_bonus_logic`，格式错误时直接减 `format_penalty_logic`；不再与 `logic_correct` 做乘法耦合。
* `R_soft` 实际是 `logic_score` 在 audit 集上的均值。
* 另外会记录 `R_soft_match_raw = mean(logic_correct_j)`（只看是否命中标准答案，不含格式奖惩），供终局回传使用。
* 这里的 `R_soft` 仍是“原始软信号”：即使当前代码尚未最终验证，逻辑命中也可以先进入 `R_code`，用于给主生成提供早期方向。
* 但“逻辑审计样本本身是否用于优化逻辑推理 token”还要经过 `confirmed` 判定：只有本节点代码已经通过，或后续被正确后代经终局回传确认，逻辑样本才会在优势计算中保留其分数。
* 逻辑/执行审计的格式判定默认允许少量标签外噪声（超参 `format_outside_noise_chars`，默认 80 个非空白字符）；设为 0 可恢复“完全严格”。

为避免公式歧义，主生成奖励改为线性相加（最后截断到 `[0,1]`）：

```python
R_code = clamp01(
    pass_rate
    + lambda_soft * R_soft_effective
    + code_compile_reward_scale * compile_score
    + code_format_reward_scale * generation_format_score
)
```

该定义保证：

* 四项贡献均可解释、可独立调参
* 在各缩放系数在 `[0,1]` 时，`R_code` 通过 `clamp01` 保持在 `[0,1]`
* 主生成奖励由三部分组成：`R_soft_effective`、`compile_score`、`generation_format_score`

### 9.4 soft reward 生效门控（与代码一致）

实现里先计算原始 `R_soft`，再做门控与缩放：

```python
soft_reward_eligible = (status_code != "SYNTAX_ERROR") and has_top_level_solve(code)
soft_scale = 1.0 if soft_reward_eligible else soft_reward_ineligible_scale  # in [0,1]
R_soft_effective = clamp01(R_soft * soft_scale)
```

也就是说：

* 节点可以有 `R_soft_raw`，在 ineligible 场景（语法错误或缺少顶层 `solve`）下，soft 奖励**降权但不必清零**。
* `soft_reward_ineligible_scale=0` 时退化为“硬门控清零”。
* 该机制在保留前期学习信号的同时，仍能抑制 reward hacking（例如输出无效代码却靠逻辑分支“刷分”）。

---

## 10. 推理奖励 R_reason（必须）

推理阶段必须拆成两段，并分别服务于不同奖励：

1. 逻辑推理分支（对应 `R_soft`）：输出 `<LOGIC_PREDICTION>`，与标准答案 `y_j` 比较。
2. 执行结果推理分支（对应 `R_reason`）：输出 `<EXEC_PREDICTION>`，与真实执行结果 `actual_j` 比较。

对 audit 集每个样例：

1. 执行代码得到 `actual_j`
2. 把 `(code, input_j)` 提供给模型生成 `<REASON>`，其中必须包含 `<EXEC_PREDICTION>`（执行推理分支不注入执行错误摘要，以避免答案线索泄漏）
3. 比较 `exec_prediction_j` 与 `actual_j`：

```python
exec_correct_j = 1 if is_match(exec_prediction_j, actual_j) else 0
R_reason = mean(exec_correct_j over audit)
```

实现细节（与代码一致）：

* 执行分支也采用独立加法格式项：

```python
if exec_format_ok:
    exec_score = clamp01(exec_correct + format_bonus_exec)
else:
    exec_score = clamp01(exec_correct - format_penalty_exec)
```

* 因此代码 / 逻辑 / 执行三条分支的格式信号都与主任务分数解耦，统一走线性加法口径。

* `status_reason=HONEST/HALLUCINATION` 的判定使用未惩罚的原始 `exec_correct_j`（raw flags）；`R_reason` 使用惩罚后的分数均值。

说明（与代码一致）：

* 不再维护节点级复合项 `R_reason_final`
* `R_reason` 只表示执行审计分支分数（执行预测正确性 + 执行格式）
* 逻辑审计格式与终局回传都在 case-level 生效（通过样本 `score/terminal_bonus/advantage` 进入训练），不再叠加到节点级综合推理奖励

### 10.1 编译/运行错误与可得分关系（与代码一致）

* 编译错误、运行错误、超时会直接影响 `pass_rate`（硬奖励下降）。
* 逻辑分支会产生 `R_soft_raw`，其计算不依赖代码可执行性；`R_code` 使用门控+缩放后的 `R_soft_effective`（见 9.4）。
* 执行分支 `R_reason` 在代码报错时也可得分：只要 `<EXEC_PREDICTION>` 正确预测了错误类型（如 `SyntaxError` / `Timeout`）。

### 10.2 终局逻辑回传奖励（与代码一致）

为了让“代码前期未通过时的逻辑信号”在终局成功后回流，加入终局回传：

```python
if descendant.pass_rate == 1.0:
    for ancestor on path(descendant -> root):
        if code_similarity(parent_code, child_code) < terminal_backprop_code_similarity_threshold:
            break
        # case-level: only audit cases with logic_correct==1 receive bonus
        for case in ancestor.logic_cases:
            if case.logic_correct == 1:
                case.terminal_bonus += terminal_logic_backprop_bonus * (terminal_logic_backprop_decay ** depth)
```

约束：

* 只对“最终有后代代码通过”的祖先生效。
* 若祖先某个 case 的逻辑预测未命中标准答案，则该 case 不回传。
* 只沿父子链回传，不跨分支。
* 同一 rollout 内，同一祖先节点只做一次回传记账（避免多个 solved 后代重复放大奖励）。
* 代码相似度门槛用于阻断“代码大改后错误归因”。
* `terminal_logic_backprop_bonus/decay/max_depth` 为可配超参。
* 终局回传命中的逻辑 case 会被标记为 `confirmed=True`；之后整题重算优势时，这些 case 才会真正为逻辑审计样本提供正向训练信号。

---

## 11. 组内相对优势（case-level）

对同一父节点兄弟组：

### 11.1 代码优势（节点级）

```python
A_code = (R_code - mean(R_code in siblings)) / (std(R_code in siblings) + eps)
```

### 11.2 逻辑优势（case-level，可跨代码）

对每个 audit case `j`：

```python
logic_case_value_j = logic_case_score_j + terminal_bonus_j if confirmed_j else 0
A_logic_case_j = zscore(logic_case_value_j over sibling nodes)
```

### 11.3 执行优势（case-level，分层）

1. 若“同代码 + 同case”样本数 >= 2：组内标准化  
2. 否则回退到 case 基线（EMA）：

```python
A_exec_case_j = exec_case_score_j - baseline_case_j
baseline_case_j <- EMA(exec_case_score_j)
```

实现约束（与代码一致）：

* 主生成仍用 `A_code`。
* 逻辑审计样本与执行审计样本都写入训练字段 `A_reason`：
  * 逻辑样本的 `A_reason` 取 `A_logic_case_j`
  * 执行样本的 `A_reason` 取 `A_exec_case_j`
* 终局逻辑回传修改 case 奖励后，会触发整题优势重算。
* 因此，“逻辑命中标准答案”本身并不自动等价于“逻辑审阅正确可学”；只有被 `confirmed` 认可后，才会真正进入逻辑审计分支的梯度。

---

## 12. 完全正确小幅更新（必须）

若节点满足：

* `pass_rate == 1` 且 `R_reason == 1`

则：

```python
A_code *= gamma_shrink
# 逻辑/执行各 case 的 advantage（最终都写入 A_reason）乘同一 shrink
A_reason_case *= gamma_shrink
```

`gamma_shrink` 必须为超参（例如 0.1），不得写死。

---

## 13. 正交损失（必须）

每个子节点的训练由三类样本构成：

```python
L = - A_code   * sum(logprob(main_generation_tokens))
    - A_reason_logic_case * sum(logprob(logic_audit_tokens))
    - A_reason_exec_case  * sum(logprob(exec_audit_tokens))
```

其中：

* `main_generation_tokens`：主生成阶段输出（单个 fenced Python code block）的全部 token
* `logic_audit_tokens`：逻辑审计阶段输出（`<REASON> + <LOGIC_PREDICTION>`）的 token
* `exec_audit_tokens`：执行审计阶段输出（`<REASON> + <EXEC_PREDICTION>`）的 token
* 不允许把两路奖励混成一个 `A_total` 更新全部 token

---

## 14. 冻结规则（必须）

每个节点评估后更新：

```python
frozen_code = (pass_rate == 1) or parent.frozen_code
frozen_reason = (frozen_code and R_reason == 1) \
             or (parent.frozen_reason and parent.frozen_code and parent.code == code)
```

后续扩展时：

* frozen_code=True → 不再生成 code（沿用该节点 code）
* 当搜索轮首次出现 `pass_rate==1` 的节点时：
  * 若 `frozen_reason_one_shot=True`，所有“通过代码但推理未冻结”的节点都会进入一次最终单轮推理阶段
  * 该阶段统一逻辑/执行审计为单路 prompt（固定代码 + 输入）
  * 每个 case 仅一次推理
  * 该阶段**不再产生主生成样本**，只产生逻辑/执行审计样本
  * 这轮结束后整题立即停止
* frozen_reason=True 且 frozen_code=True → 该节点本身已完全冻结
* **若代码发生变化（未冻结或被改写），则必须重算推理审计，不可复用旧 `R_reason`**

---

## 15. 重采样门控（必须）

对兄弟组 (\mathcal{S})：

若对所有子节点同时满足：

```python
R_code == 0 and R_reason == 0
```

则：

* 丢弃该兄弟组
* 重试生成
* 重试最多 `N` 次（实现参数名：`M_retry`）
* 每次重试生成的节点也计入 Node_Count
* 若最后一次重试后仍“全双0”，则不再继续重试，保留最后一次兄弟组（随后可被节点级双0规则拦截）

### 15.1 剪枝触发条件（必须明确）

实现里的“剪枝/不再扩展”包含四类：

1. **节点级双0剪枝**：当节点满足 `R_code==0 and R_reason==0`，该节点不进入下一轮 frontier。
2. **兄弟组级重采样**：当某次扩展生成的 sibling 组满足“全体双0”，整组丢弃并按 `M_retry` 重试。
3. **冻结剪枝**：当节点 `frozen_code=True and frozen_reason=True` 时，该节点不再进入下一轮 frontier。
4. **代码通过后停止搜索**：一旦某轮出现通过代码节点，未通过节点不再继续扩展；随后进入最终单轮推理阶段或直接停止。
5. **题目级终止剪枝**：命中停止条件（见第 16 节）后，该题停止继续扩展。

---

## 16. 停止条件（必须）

对单个问题，代码搜索阶段停止当任一满足：

1. 当前搜索轮已出现至少一个 `pass_rate==1` 的节点
2. `Round > T_max`
3. `Node_Count >= N_max`
4. frontier 为空

若满足第 1 条且 `frozen_reason_one_shot=True`，则额外执行一次最终单轮推理阶段，然后整题停止。

---

## 17. 测试模式（必须跑多轮 + 反馈驱动）

测试模式要求：

* 允许最多 `T_max` 轮
* 每轮可以把“父代码摘要 + 执行错误反馈 + 失败样例摘要”加入上下文（受 `context_round_window` 与截断规则约束）
* 不计算/不反传梯度，不调用 `optimizer.step()`
* 但仍应计算 pass_rate 与（可选）reason_score 以便调试（**指标统计只看 code**，见下一节）

---

## 18. 评估指标（可配置 n 与 k；以 code 为准）

### 18.1 指标参数

* `eval_round_n`：统计第 n 轮结果（n 可配置）
* `eval_k_list`：例如 `[1,3,5]`（k 可配置）

### 18.2 pass@k@round=n（你的要求）

定义：只统计第 n 轮生成的候选代码中，前 k 个是否有一个全过：

```python
pass_at_k_round_n = (#题目中第n轮前k个候选里存在pass_rate==1) / 总题目数
```

### 18.3 额外允许的 code-only 指标（可选）

* `pass@k`（把 1..n 轮的候选按生成顺序拼接取前 k 个）
* `best_pass_rate@round=n`（第 n 轮中最大 pass_rate 的平均）
* `best_pass_rate_overall`（所有轮次中最大 pass_rate 的平均）

> **评估指标系统必须以 code 为主，不要求 reason 数值参与指标。**
> （reason 可以作为调试信息输出，但不进入 pass@k 指标。）

---

## 19. 调试可视化输出（必须，尤其 test）

在 test 模式必须输出 JSON（可配置只保存少量节点以控体积）：

```python
{
  "question_id": ...,
  "audit_indices": [...],
  "rounds": [
    {
      "round": 1,
      "nodes": [
        {
          "node_id": ...,
          "parent_id": ...,
          "pass_rate": ...,
          "status_code": ...,
          "R_code": ...,
          "R_reason": ...,
          "R_soft_raw": ...,
          "R_soft_match_raw": ...,
          "R_soft_effective": ...,
          "soft_reward_eligible": ...,
          "terminal_backprop_bonus": ...,
          "frozen_code": ...,
          "frozen_reason": ...,
          "pruned_double_zero": ...,
          "generation_debug": {
            "raw_output": ...,
            "parsed_reason": ...,
            "parsed_logic_prediction": ...,
            "parsed_exec_prediction": ...,
          },
          "logic_audit": [
            {
              "case_index": ...,
              "input": ...,
              "expected_output": ...,
              "raw_output": ...,
              "parsed_prediction": ...,
              "format_ok": ...,
              "match": ...,
              "score_after_penalty": ...,
            }
          ],
          "exec_audit": [
            {
              "case_index": ...,
              "input": ...,
              "actual_kind": ...,
              "actual_value": ...,
              "actual_error_type": ...,
              "actual_error_msg": ...,
              "raw_output": ...,
              "parsed_prediction": ...,
              "format_ok": ...,
              "match": ...,
              "score_after_penalty": ...,
            }
          ],
          "code_preview": ...,
          "error_summary": ...,
        }
      ]
    }
  ]
}
```

---

## 20. 日志与面板（必须）

### 20.1 日志

* 必须复用现有 logging 体系
* 不允许 `print`
* 新增分类（或 tag）至少包括：`TREE, REWARD, TRAIN, TEST, EVAL`
* 必须落盘纯文本日志（`txt`），至少包含：
  * 运行级日志：`runtime_rank*.txt`
  * 训练/评估事件日志：`trainer_events_rank*.txt`
* 结构化事件日志（`jsonl`）可与 `txt` 并存，但不能替代 `txt`

### 20.2 TensorBoard（服务器远程查看）

必须记录（至少）：

* `loss_code`, `loss_reason`
* `mean_R_code`, `mean_R_reason`
* `mean_R_soft_raw`, `mean_R_soft_match_raw`, `mean_R_soft_effective`, `soft_reward_eligible_rate`
* `mean_terminal_backprop_bonus`
* `mean_pass_rate`
* `std_R_code`, `std_R_reason`
* `node_count`, `final_reason_node_count`, `resample_count`
* `pass_at_k_round_n`（按配置）

说明（与当前实现一致）：

* 未带前缀的主指标默认按**搜索阶段节点**统计，不把 `final_reason` 节点混入 `mean_R_code / mean_pass_rate / generation_format_ok_rate` 等主监控。
* 最终单轮推理阶段单独记录为 `final_reason_*` 指标，例如：
  * `final_reason_mean_R_reason`
  * `final_reason_logic_sample_count`
  * `final_reason_exec_sample_count`

### 20.3 产物目录与命名（必须）

所有产物必须按 run 归档，不允许散落在当前目录。目录结构要求：

```text
<base_output_dir>/
  <mode>/                                      # train 或 test
    <YYYYMMDD_HHMMSS>__<mode>__<model>__<dataset>__<backend>/
      run_manifest.json                        # 本次运行配置快照 + 路径索引
      logs/
        runtime_rank0.txt
        trainer_events_rank0.txt
        trainer_events_rank0.jsonl
        rollout_summary_rank0.jsonl
        test_metrics.txt                       # 仅 test 模式
      tensorboard/                             # report_to=tensorboard 时生效
      train_out/ 或 test_out/
        checkpoint-*/
        trainer_state.json
        traces/rollout/*.json                  # debug_trace_dir 默认值
```

命名字段要求：

* `timestamp`：启动时间（本地时区）
* `mode`：`train|test`
* `model`：模型名短标识（去路径）
* `dataset`：数据集短标识（dataset_name 或 data_files 推导）
* `backend`：`hf|vllm`

这样可以直接从路径判断“何时生成、基于什么模型/数据/后端、属于 train 还是 test”。

---

## 21. 最小修改原则（必须）

在现有开源代码上修改时：

* 禁止重写核心训练主结构（Trainer 主循环/模型封装/优化器定义）
* 只能通过：

  * 新增模块（extensions / algo / adapters / backend）
  * wrapper / hook / callback 注入
  * 配置驱动切换
* 修改现有文件必须做到：

  * 只加最少调用点
  * 不改原有接口签名
  * 不破坏原功能

---

## 22. 超参数集中管理（必须）

以下参数必须集中配置（yaml/cli 二选一或同时支持）：

* `K, K_reason, T_max, N_max, M_audit, M_retry`
* `context_round_window`
* `lambda_soft, code_compile_reward_scale, code_format_reward_scale, soft_reward_ineligible_scale, beta_reason, gamma_shrink`
* `logic_format_reward_scale`（兼容旧配置保留；当前实现中不参与节点级 `R_reason` 计算）
* `format_penalty_logic, format_bonus_logic, format_penalty_exec, format_bonus_exec`
* `use_chat_template_for_codegrpo`
* `terminal_logic_backprop_bonus, terminal_logic_backprop_decay, terminal_logic_backprop_max_depth`
* `terminal_backprop_code_similarity_threshold`
* `exec_case_baseline_ema_alpha`
* `frozen_reason_one_shot, unify_reason_when_code_frozen`
* `reasoning_max_chars, prediction_max_chars, format_outside_noise_chars`
* `max_completion_length, max_completion_length_code, max_completion_length_audit`
* `error_max_chars, error_max_lines`
* `eval_round_n, eval_k_list`
* backend 选择：`backend=hf|vllm`

长度约束：

* `max_completion_length` 是共享后端的全局上限。
* `max_completion_length_code` 是主生成代码的上限；若为空则回退到 `max_completion_length`。
* `max_completion_length_audit` 是逻辑/执行审阅以及 final-reason 统一审阅的上限；若为空则回退到 `max_completion_length`。
* 约束关系：`max_completion_length >= max(max_completion_length_code, max_completion_length_audit)`。

禁止在代码中散落硬编码常量。

---

### 最后：实现时必须特别注意的“易错点清单”

（这部分也是约束）

1. **评估指标只看 code**：pass@k / pass@k@round=n 不依赖推理
2. **测试模式也允许多轮**：用错误反馈推进；但不更新参数
3. **sibling 必须共享 audit**：同父扩展的 K=2 必须用同一组 audit 样本
4. **soft reward 来自逻辑推理分支**：`<LOGIC_PREDICTION>` 与标准答案比较；`R_code` 采用线性相加（pass/soft/compile/format）
5. **GRPO 标准化**：`A_code` 节点级；`A_logic/A_exec` 按 case-level 计算
6. **输出匹配要鲁棒**：结构化解析优先，错误类型匹配必须支持
7. **节点级双0剪枝 + 组级全双0重采样**：两者同时生效，且节点预算计数包含重采样与剪枝
8. **code/reason 正交更新**：不得用一个总优势更新所有 token
9. **完全正确缩放**：必须对 `A_code` 与各推理样本写入 `A_reason` 的 case-level 优势同时 shrink
10. **终局回传需做代码一致性门控**：代码相似度低于阈值必须停止回传
11. **终局回传后必须重算优势**：否则 case-level 优势与新奖励不一致
12. **三类生成统一走 chat template（默认）**：避免 instruct 模型输出格式漂移

---

## 23. 逐轮演算示例（对照实现）

设定：

* `K=2, T_max=3, N_max=8, M_retry=1, M_audit=2`
* `lambda_soft=0.2`
* `soft_reward_ineligible_scale=0.3`
* `K=2, K_reason=4`
* `R_code = clamp01(pass_rate + lambda_soft*R_soft_effective + code_compile_reward_scale*compile_score + code_format_reward_scale*generation_format_score)`

题目：修复某函数，测试集 4 条。

### Round 1（parent=root）

1. 首次扩展生成 `n1,n2`（共 2 节点，Node_Count=2）。
2. 审计后：
   * `n1`: `pass_rate=0, R_soft=0, R_reason=0` → `R_code=0`，双0
   * `n2`: `pass_rate=0, R_soft=0, R_reason=0` → `R_code=0`，双0
3. 因为同组全双0，触发组级重采样（retry=1）。
4. 再生成 `n3,n4`（Node_Count=4，重采样节点也计数）：
   * `n3`: `pass_rate=1.0, R_soft=0.0, R_reason=0.0` → `R_code=1.0`
   * `n4`: `pass_rate=0.5, R_soft=0.5, R_reason=0.5`  
     `R_code = clamp01(0.5 + 0.2*0.5 + 0.1*1 + 0.08*1) = 0.78`
5. 节点级剪枝：`n3,n4` 都不是双0，先进入候选 frontier。
6. 由于出现了 `pass_rate==1` 的 `n3`，代码搜索阶段在本轮后停止。
7. 冻结状态：
   * `n3`: `frozen_code=True`（代码全过），`frozen_reason=False`
   * `n4`: `frozen_code=False, frozen_reason=False`

### Final Reason Round（from passed nodes）

#### 从 `n3` 扩展

1. 因 `n3.frozen_code=True` 且 `frozen_reason_one_shot=True`，进入最终单轮推理阶段，只修 reason。
2. 本轮按 `K_reason=4` 生成 `n5,n6,n7,n8`，每个 audit case 只生成一次统一推理（固定代码+输入）。
3. 这 4 个节点记入 `final_reason_node_count`，不计入搜索阶段 `Node_Count`，也不受 `N_max/T_max` 截断。
4. `A_logic/A_exec` 按 case-level 计算：逻辑可跨代码 sibling 比较；执行优先同代码同case，样本不足回退 EMA baseline。
5. 这轮结束后整题直接停止，不再继续扩树。

### 终止判断

* 若 `n5` 满足 `pass_rate=1 and R_reason=1`，则该节点已完全冻结。
* 即使搜索阶段此时已达到 `T_max` 或 `N_max`，这次最终单轮推理仍会完整执行。

### 这个演算对应的关键行为

1. “2 变 4”有两种来源：首组全双0触发重采样；或代码通过后进入 `K_reason=4` 的最终单轮推理阶段。
2. 双0节点会记录在 trace 里，但不进入下一轮 frontier。
3. 代码通过后会停止代码搜索；若启用 `frozen_reason_one_shot=True`，所有通过代码节点都会各自执行一次 one-shot reason-only 轮。
4. `A_logic` 与 `A_exec` 分开：逻辑按 case-level sibling 比较；执行优先同代码同case，不足时回退 case EMA。
5. `R_code` 为线性可解释加和（pass/soft/compile/format），再 `clamp01`。
6. 最终单轮推理节点单独统计，不计入搜索阶段 `Node_Count`。




## Eval Mode

Primary evaluation uses a code-only single trajectory.

- No logic audit is run during eval.
- No execution audit is run during eval.
- No sibling tree is expanded during eval.
- Each eval sample follows exactly one repair path:
  - round 1: generate one code candidate
  - round 2: if still unsolved and `T_max >= 2`, generate one repaired code candidate
  - ...
  - stop early once the code passes all tests

Eval therefore reports cumulative `pass@1` within `<= r` rounds for every `r <= T_max`:

- `pass_at_1_round_1`: solved within 1 round
- `pass_at_1_round_2`: solved within 2 rounds
- ...
- `pass_at_1_round_T_max`: solved within `T_max` rounds

`pass_at_1` is the same as `pass_at_1_round_T_max`.

The old full-rollout eval with audit branches is no longer the primary metric path.
