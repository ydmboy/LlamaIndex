# 项目问题诊断与行动计划

> 诊断时间：2026-07-28
> 使用方式：请在本文档上直接修改/标注，告诉我执行哪些项（可勾选 [x] 或直接批注）。
> 注：所有 git 变更类操作（add/commit/stash drop）执行前我会再与你确认一次。

---

## 现状诊断摘要

仓库卡在 `git stash pop` 后的未收尾状态：`run_vector_demo.py` 为 `UU`（未解决的合并冲突），但冲突内容已在工作区手工合并完成，只差 `git add` 标记解决并提交。

已验证的事实（2026-07-28 实测）：

- 工作区 `run_vector_demo.py`（1962 行）无冲突标记，`ast.parse` 通过、import 实测通过
- 冲突双方改动均已保留：
  - HEAD 侧：`retrieval_config.yaml` 统一检索参数（`_load_retrieval_config` / `_make_query_engine` / `_RETR_CFG`）
  - stash 侧：流式持久化补丁（`_streaming_kvstore_persist`，防大库持久化内存爆炸）、ingest 分批（`_input_with_timeout`）
- `app.py` 从 `run_vector_demo` 导入的 18 个符号全部存在
- `stash@{0}`（WIP on main: bc78726）仍在，是当前的安全网，确认提交无误前**不要 drop**

当前 git 状态：

```
UU run_vector_demo.py      ← 未标记解决的合并冲突（内容已合好）
MM MANUAL.md               ← 部分改动已暂存、部分未暂存
 M retrieval_config.yaml   ← 未暂存（top_k 5→50，新增 ingest 分区）
stash@{0}                  ← 冲突来源，安全网
```

---

## P0 — 合并冲突收尾（建议最先执行）

- [ ] **P0-1** `git add run_vector_demo.py`，标记冲突已解决
- [ ] **P0-2** 提交合并结果（commit message 建议：`merge: 合并检索参数配置与流式持久化补丁（stash pop 冲突解决）`，可按你的习惯改）
- [ ] **P0-3** 确认提交无误后 `git stash drop stash@{0}`，清理安全网

## P1 — 参数一致性确认（提交前需要你拍板）

- [ ] **P1-1** `retrieval_config.yaml` 中 `similarity_top_k: 50`（未暂存改动，原值 5）
  - 影响：配合 `tree_summarize`，50 个 chunk × 512 字符 ≈ 2.5 万 token/轮塞进 LLM，DeepSeek 调用变慢变贵
  - 待确认：是有意调大还是调试残留？→ 保留 50 / 改回 5 / 其他值：____
- [ ] **P1-2** `ingest.batch_size` 代码默认值 1000 vs yaml 值 2000 不一致
  - 不影响运行（yaml 覆盖默认值），但建议统一
  - 待确认：统一为哪个值？____，改代码默认还是改 yaml？____

## P2 — 文档同步（可在 P0/P1 之后做）

- [ ] **P2-1** 更新 `PROJECT_OVERVIEW.md`：行数 1465→1962、补充 `app.py`（Web UI）、`retrieval_config.yaml`、ingest 分批、流式持久化补丁；修正 `models/` 目录描述（实际只剩 `bge-m3`）
- [ ] **P2-2** `MANUAL.md` 的暂存/未暂存改动一起检查后随合并一并提交

---

## 执行记录（2026-07-28 已全部执行完毕）

| 项 | 状态 | 备注 |
|----|------|------|
| P0-1 | ✅ 已执行 | `git add run_vector_demo.py`，冲突标记解决 |
| P0-2 | ✅ 已执行 | 提交 `fb43125`（merge: 解决 stash pop 冲突…） |
| P0-3 | ✅ 已执行 | 验证 import/参数通过后 `stash drop`，stash 已清空 |
| P1-1 | ✅ 已拍板执行 | 50→**10**（提交 `6bbc6dc`）。依据：无 rerank 的 RAG 实践取值 5~10；MANUAL 本就按 10 描述，50 是测试残留 |
| P1-2 | ✅ 已拍板执行 | 统一为 **1000**（改 yaml，提交 `6bbc6dc`）。依据：与代码兜底默认及 MANUAL「每 1000 文件落盘检查点」一致，单一真源 |
| P2-1 | ✅ 已执行 | PROJECT_OVERVIEW.md 全面重写同步（提交 `026af88`），34 个函数行号按 ast 实测 |
| P2-2 | ✅ 已执行 | MANUAL.md 暂存+未暂存改动分别随 `fb43125`/`6bbc6dc` 提交，并补了 ingest 分区参数表 |

最终状态：`main` 领先 `origin/main` 3 个提交，工作区干净（仅本文件未跟踪）。**尚未 push**，需要时执行 `git push`。
