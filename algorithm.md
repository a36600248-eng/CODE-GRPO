## 0. 术语与符号（必须统一）

* **问题样本**：一个编程题 + 测试集
* **测试集**：(\mathcal{D}={(x_j, y_j)}_{j=1}^N)
* **审计子集（audit）**：从 (\mathcal{D}) 固定抽取 (M_{audit}) 个样例，用于推理一致性与 soft reward
* **轮次（round）**：同一问题内的多轮修复/生成，最多 (T_{max}) 轮
* **兄弟组（sibling group）**：同一父节点一次扩展生成的 (K) 个子节点（这里 (K=2)）
* **节点预算**：同一问题允许生成的节点总数上限 (N_{max})
* **节点计数口径**：每生成一个“候选解”（包含 code/reason 两段之一或两段）计 1；被剪枝/重采样的也计入
* **模式**：`mode=train|test`（两者都允许多轮；test 不更新参数）

---

## 0.1 一眼看懂流程（建议先读）

1. 每个父节点每轮扩展 `K` 个子节点。
2. 每个子节点都计算两类奖励：`R_code` 与 `R_reason`。
3. **节点级剪枝**：若某子节点满足 `R_code==0 and R_reason==0`，该节点不进入下一轮。
4. **组级重采样**：若同一父节点扩展出的整组 sibling 全部“双0”，则整组丢弃并重采样，最多重试 `N` 次（实现参数名 `M_retry`）。若达到重试上限仍为“双0”，实现会保留最后一次生成结果（随后仍会被节点级双0剪枝）。
5. 训练时对保留样本计算正交 loss；测试时不更新参数，只统计指标和输出 trace。
6. 节点计数始终包含：正常生成、被剪枝节点、重采样节点。

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
* 历史内容不只含错误反馈，还应含：
  * 上一轮（或最近 W 轮）代码片段摘要（用于“在已有代码上修复”而不是每轮重写）
  * 失败样例摘要（`failed_input / failed_actual`）
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
  R_reason_final: float

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

生成输出必须可可靠解析为：

```text
<CODE>
...
</CODE>
<REASON>
...（思考过程）
<LOGIC_PREDICTION>
...（基于代码意图与输入，预测理想输出；忽略语法错误等实现小问题）
</LOGIC_PREDICTION>
<EXEC_PREDICTION>
...（基于代码与输入，预测实际执行结果：输出或错误类型）
</EXEC_PREDICTION>
</REASON>
```

### 7.2 token mask 必须可得

实现必须能得到：

* `code_token_mask`
* `reason_token_mask`

并保证：

* code 的奖励只作用于 code tokens
* reason 的奖励只作用于 reason tokens

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
logic_score = max(0, logic_correct - format_penalty_logic)
```

* 合法格式时 `logic_score = logic_correct`。
* `R_soft` 实际是 `logic_score` 在 audit 集上的均值。

为避免 soft reward 掩盖硬奖励（pass_rate），代码奖励按“余量缩放”定义：

```python
R_code = pass_rate + lambda_soft * (1 - pass_rate) * R_soft
```

该定义保证：

* `R_code >= pass_rate`
* 当 `pass_rate` 越高，soft reward 的增益越小
* 在 `R_soft in [0,1]` 且 `lambda_soft <= 1` 时，`R_code <= 1`

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

* 执行分支也有格式惩罚 `format_penalty_exec`；格式不合法时分数会被扣减并截断到 `[0,1]`。
* `status_reason=HONEST/HALLUCINATION` 的判定使用未惩罚的原始 `exec_correct_j`（raw flags）；`R_reason` 使用惩罚后的分数均值。

调制（必须）：

```python
R_reason_final = R_reason * (0.5 + 0.5 * pass_rate)
```

### 10.1 编译/运行错误与可得分关系（与代码一致）

* 编译错误、运行错误、超时会直接影响 `pass_rate`（硬奖励下降）。
* 逻辑分支 `R_soft` 不依赖代码可执行性；即使代码不可执行，只要逻辑预测接近标准答案，仍可能获得 soft reward。
* 执行分支 `R_reason` 在代码报错时也可得分：只要 `<EXEC_PREDICTION>` 正确预测了错误类型（如 `SyntaxError` / `Timeout`）。

---

## 11. 组内相对优势（GRPO 标准化，必须除以标准差）

对同一父节点生成的兄弟组 (\mathcal{S})（K=2）：

### 11.1 组内均值与方差

```python
mean_code = mean(R_code in group)
std_code  = std(R_code in group)

mean_reason = mean(R_reason_final in group)
std_reason  = std(R_reason_final in group)
```

### 11.2 优势（必须标准化）

```python
A_code   = (R_code - mean_code) / (std_code + eps)
A_reason = (R_reason_final - mean_reason) / (std_reason + eps)
```

> 这条是你新增要求：**与 GRPO 一致，必须除以 std。**

---

## 12. 完全正确小幅更新（必须）

若节点满足：

* `pass_rate == 1` 且 `R_reason == 1`

则：

```python
A_code *= gamma_shrink
A_reason *= gamma_shrink
```

`gamma_shrink` 必须为超参（例如 0.1），不得写死。

---

## 13. 正交损失（必须）

每个子节点的 loss：

```python
L = - A_code   * sum(logprob(code_tokens))
    - A_reason * sum(logprob(reason_tokens))
```

* `logprob(code_tokens)` 只对 code mask 生效
* `logprob(reason_tokens)` 只对 reason mask 生效
* 不允许混合成一个 A_total 去更新全部 tokens

---

## 14. 冻结规则（必须）

每个节点评估后更新：

```python
if pass_rate == 1:
    frozen_code = True
if R_reason == 1:
    frozen_reason = True
```

下一轮生成时：

* frozen_code=True → 不再生成 code（沿用该节点 code）
* frozen_reason=True → 不再生成 reason（沿用该节点 reason 或直接跳过 reason 生成）

若两者都 True → 该问题停止多轮。

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

实现里的“剪枝/不再扩展”包含三类：

1. **节点级双0剪枝**：当节点满足 `R_code==0 and R_reason==0`，该节点不进入下一轮 frontier。
2. **兄弟组级重采样**：当某次扩展生成的 sibling 组满足“全体双0”，整组丢弃并按 `M_retry` 重试。
3. **冻结剪枝**：当节点 `frozen_code=True and frozen_reason=True` 时，该节点不再进入下一轮 frontier。
4. **题目级终止剪枝**：命中停止条件（见第 16 节）后，该题停止继续扩展。

---

## 16. 停止条件（必须）

对单个问题停止，当任一满足：

1. 存在节点：`pass_rate==1 and R_reason==1`
2. `Round > T_max`
3. `Node_Count >= N_max`

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

### 20.2 TensorBoard（服务器远程查看）

必须记录（至少）：

* `loss_code`, `loss_reason`
* `mean_R_code`, `mean_R_reason_final`
* `mean_pass_rate`
* `std_R_code`, `std_R_reason_final`
* `node_count`, `resample_count`
* `pass_at_k_round_n`（按配置）

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

* `K, T_max, N_max, M_audit, M_retry`
* `context_round_window`
* `lambda_soft, beta_reason, gamma_shrink`
* `error_max_chars, error_max_lines`
* `eval_round_n, eval_k_list`
* backend 选择：`backend=hf|vllm`

禁止在代码中散落硬编码常量。

---

### 最后：实现时必须特别注意的“易错点清单”

（这部分也是约束）

1. **评估指标只看 code**：pass@k / pass@k@round=n 不依赖推理
2. **测试模式也允许多轮**：用错误反馈推进；但不更新参数
3. **sibling 必须共享 audit**：同父扩展的 K=2 必须用同一组 audit 样本
4. **soft reward 来自逻辑推理分支**：`<LOGIC_PREDICTION>` 要与标准答案比较；`R_code = pass_rate + lambda_soft * (1 - pass_rate) * R_soft`
5. **GRPO 标准化**：优势必须除以组内 std（+eps）
6. **输出匹配要鲁棒**：结构化解析优先，错误类型匹配必须支持
7. **节点级双0剪枝 + 组级全双0重采样**：两者同时生效，且节点预算计数包含重采样与剪枝
8. **code/reason 正交更新**：不得用一个总优势更新所有 token
9. **完全正确缩放**：必须对 A_code 与 A_reason 同时 shrink

---

算法总体流程浏览例子：
假设每次生成2个样本。某父节点扩展出子节点 A、B：
* 若 A 满足 `R_code==0 and R_reason==0`，则 A 被节点级剪枝，不进入下一轮；
* 若 B 不是双0，则 B 进入下一轮并继续扩展为 2 个新子节点。

如果某父节点扩展出的整组 sibling（例如 A、B）全是双0，则整组丢弃并重采样，最多重试 `N` 次（实现参数 `M_retry`）。若达到重试上限仍全双0，保留最后一组（随后通常被节点级双0规则拦截）。无论是被剪枝节点还是重采样节点，都计入 `Node_Count`。




