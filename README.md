# SynAgentics ADC Backend (Step 1-14 MVP Skeleton)

后端骨架，对应 `项目文件/SynAgentics_ADC_Backend_Architecture_Design_v0.1.md` 与
`项目文件/ADC_Pipeline_IO_Schema_v0.1.md`。当前阶段为骨架交付：

- 目录与模块完整
- pydantic schema 字段已对齐 IO Schema
- `intake / input_readiness / workflow_setup` 三个确定性 service 已最小可跑通
- 本地 storage adapter 可读写
- Step 5-9 / 13 / 14 LLM agent、真实 MCP 远端调用、Step 10 外部 AEE 暂未实现

## 启动

```bash
# 本地开发：把 .env.example 复制为 .env，再填入真实 API key。
# .env 已在 .gitignore 中，不要提交到 GitHub。
# 不需要真实 key 时保持空值即可 —— 默认 LLM_PROVIDER=mock 不读 GEMINI_API_KEY。
cp .env.example .env
pip install -e ".[dev]"

# 本地跑 Step 1→4 链路（写到 ./.localstore/...）
STORAGE_MODE=local QUEUE_MODE=memory python scripts/seed_local_run.py

# 启动 API
uvicorn app.main:app --reload
# 文档：http://localhost:8000/docs
```

## Step 1 — Run ingestion 入口

Step 1 现在有两种 ingestion 入口，**都**最终走 `IntakeService.submit(...)`，产出同一种
`raw_request_record`，只是文件来源不同。

### A) JSON internal ingestion — `POST /runs`

```bash
curl -X POST http://localhost:8000/runs -H 'Content-Type: application/json' -d '{
  "raw_user_query": "Design ADC against HER2 with vc-MMAE",
  "user_provided_context": {
    "target_or_antigen_text": "HER2",
    "candidate_text": "Trastuzumab analog",
    "payload_linker_text": "vc-MMAE"
  }
}'
```

适用：notebook、脚本、内部 worker。前面不接前端时的默认入口。
不上传文件。

### B) 模拟前端的 multipart 入口 — `POST /runs/multipart`

适用：模拟前端用 `multipart/form-data` 同时发自然语言 + 一个或多个文件
（pdb / cif / fasta / csv …）。

```bash
curl -X POST http://localhost:8000/runs/multipart \
  -F 'raw_user_query=Design ADC against HER2 with vc-MMAE' \
  -F 'entry_source=ui' \
  -F 'submitted_by=tester@example.com' \
  -F 'user_provided_context={"target_or_antigen_text":"HER2","candidate_text":"Trastuzumab","payload_linker_text":"vc-MMAE"}' \
  -F 'files=@/path/to/complex.pdb;type=chemical/x-pdb' \
  -F 'files=@/path/to/heavy_chain.fasta;type=text/x-fasta'
```

- `user_provided_context` 以 **JSON string** 传入，后端解析；**必须**是 JSON object，array/标量返回 422。
- `files` 字段可重复，每个 part 是一个文件。

### 上传限制与失败行为（服务端强制）

后端不依赖前端校验，所有限制由 `app/api/step_01_intake_multipart_api.py` 在写文件**之前**和读取过程中强制：

| 校验项 | 行为 | HTTP |
|---|---|---|
| `raw_user_query` 为空 / 全空白 | 拒绝，无任何文件写入 | 422 |
| `entry_source` 不在 `{ui, api, notebook, script}` | 拒绝，无任何文件写入 | 422 |
| `user_provided_context` 非合法 JSON 或非 object | 拒绝，无任何文件写入 | 422 |
| 文件数量 > `MAX_UPLOAD_FILES_PER_RUN`（默认 10） | 拒绝，无任何文件写入 | 413 |
| 单文件字节数 > `MAX_UPLOAD_BYTES_PER_FILE`（默认 50 MiB） | 拒绝，该请求**之前**已写入的所有文件被清理 | 413 |
| `IntakeService.submit(...)` 在文件已写入后抛错 | 该请求所有已写入文件被清理，registry / workflow_state 不留半成品 | 500 |

可通过 `.env`（或环境变量）调节：

```
MAX_UPLOAD_FILES_PER_RUN=10
MAX_UPLOAD_BYTES_PER_FILE=52428800
```

**失败清理策略**：每个 multipart 请求在 handler 内部维护一份 `written_keys` 列表。任何抛出 `HTTPException` 或其它异常的分支都会先 `storage.delete(key)` 清理所有已落盘 key，再让异常向上传播。结果：**要么整个请求成功并产出 raw_request_record + 所有文件，要么磁盘上不留任何这次请求的 byte 或 artifact**。

### 文件落盘

multipart 入口在 Step 1 把每个上传文件**真实落盘**到当前 run 目录下：

```
{prefix}/runs/{run_id}/
  inputs/
    files/{file_id}{ext}        ← 原始文件字节
    raw_request_record.json     ← 业务 artifact，仅元数据
    structured_query.json
    input_readiness_status.json
    run_step_plan.json
  candidate_context_table.json
  structured_liability_summary.json
  tool_outputs/step_05/...
  tool_outputs/step_06/...
```

每个文件在 `raw_request_record.uploaded_files[]` 里只放
`file_id / original_filename / storage_path / content_type / sha256 / size_bytes`
等**元数据**；原始字节**只**存在 `inputs/files/`，**不**进 JSON artifact。

`storage_path` 是 **storage key**（如 `adc_pilot/runs/{run_id}/inputs/files/file_xxx.pdb`），
不是绝对文件路径，也不是文件正文。读字节请用 `storage.read_bytes(storage_path)`；
切到 S3 后同一 key 直接生效。

### 下游消费

- Step 3 `InputReadinessService` 用 `uploaded_files[].original_filename + content_type`
  推断角色：`pdb_or_cif_structure / fasta_sequence / csv_or_table / json_metadata /
  image / unknown`。
- Step 5 `CandidateContextAgent` 把 `*.pdb/.cif` 转成 target candidate 的
  `structure_file` material，把 `*.fasta` 转成 antibody candidate 的
  `antibody_heavy_chain_sequence` material。
- Step 6 / 7-9 lane router 通过 material_type 判断是否有 `structure_file /
  structure_ref` 决定 structure lane 是否激活。

## Step 13-14 — EvidenceAgent 与 PatentIPAgent

Step 13 是**scientific evidence search**；Step 14 是 **patent / prior-art / regulatory reference scan**。两个 agent 都 deterministic 构造 query（**不调 LLM**），通过 inventory-scoped `MCPClient` 发起 MCP 调用。

| Step | agent | 主要 MCP tools (v0.2 inventory) | 输出 artifact |
|---|---|---|---|
| 13 | `EvidenceAgent` | `EuropePMC_search_articles`, `SemanticScholar_search_papers`, `LiteratureSearchTool`, `PubTator3_LiteratureSearch`, `MultiAgentLiteratureSearch`（可选） | `scientific_evidence_table.json` |
| 14 | `PatentIPAgent` | `PubChem_get_associated_patents_by_CID`, `drugbank_get_drug_references_by_drug_name_or_id`, `FDA_OrangeBook_get_patent_info` | `patent_prior_art_table.json` |

**Upstream artifacts 与 fallback 顺序**：

Step 13 EvidenceAgent 显式读取：
- Step 2 `structured_query` —— `mentioned_entities.target_or_antigen_text` / `payload_text` / `linker_text`
- Step 5 `candidate_context_table` —— per-candidate `candidate_label`（用于 `PubTator3_LiteratureSearch`）
- Step 10 `scoring_handoff_package` —— `candidate_ids` 用作 shortlist fallback（**新增**）
- Step 12 `ranking_table` —— `ranked_candidates` 用作 shortlist（最优先）

Step 13 shortlist 解析顺序（在 `MultiAgentLiteratureSearch.tool_input_summary.shortlist_source` 中如实标注）：
1. `step_12_ranking` —— `ranking_status="completed"` 且 `ranked_candidates` 非空时
2. `step_10_handoff` —— Step 10 handoff 存在且 `candidate_ids` 非空时（Step 11/12 awaiting 路径走这里）
3. `step_05_candidates` —— 最终回退

Step 14 PatentIPAgent 显式读取：
- Step 2 `structured_query` —— `mentioned_entities.payload_text` / `linker_text` 作为 text-only fallback（**新增**）
- Step 5 `candidate_context_table` —— `compound_component` candidate 的 `payload_name`/`linker_name`/`compound_name` material + `pubchem_cid` identifier
- Step 9 `compound_screening_artifact` —— compound hits 决定是否需要 text-only fallback
- Step 10 `scoring_handoff_package` —— `candidate_ids` 用作 scope fallback（**新增**）
- Step 12 `ranking_table` —— `ranked_candidates` 用作 scope（最优先，**新增**）

Step 14 scope 解析顺序（在每次 tool call 的 `tool_input_summary.shortlist_source` 中如实标注）：
1. `step_12_ranking` —— 完成且非空时，**只**对 ranked candidate 发起 PubChem / DrugBank / Orange Book 查询
2. `step_10_handoff` —— Step 10 candidate_ids 非空时
3. `step_05_candidates` —— Step 5 compound candidate 全量
4. `step_02_structured_query` —— 当上述三层都没有 compound candidate 命中（Step 5 没有 compound_component、Step 9 没有 compound_hits）但 structured_query 仍提到 payload/linker 文本时，对该文本发起一次 DrugBank + Orange Book 查询，shortlist_source 标注为 `step_02_structured_query`，对应 `patent_records[].candidate_id=""`

没有 ranking 也能跑：Step 13 走 handoff → Step 5 candidates；Step 14 走 handoff → Step 5/9 → Step 2 payload 文本，**不 crash**。

**Raw 落地**：
- Step 13 → `tool_outputs/step_13/{tool_call_id}.json`
- Step 14 → `tool_outputs/step_14/{tool_call_id}.json`

**Normalized artifact 只保留**：
- Step 13：`evidence_records[]`（`evidence_id / candidate_id / target / mechanism / evidence_type / key_finding / source / confidence_score`）+ `tool_call_records[].tool_output_ref` + `review_status`
- Step 14：`patent_records[]`（`patent_record_id / candidate_id / matched_entity_type / source_database / source_ref / notes_limitations / 等`）+ `tool_call_records[].tool_output_ref` + `legal_disclaimer` + `patent_review_status`

**FDA Orange Book 处理**：source_database 一律写 `FDA_OrangeBook`，matched_entity_type 写 `drug_application_or_regulatory_reference`；产品/专利/exclusivity 原始行**只在** raw payload（tool_output_ref）里，**不进** `patent_records[]`。

**`legal_disclaimer`**：artifact 顶级始终携带 "For demonstration purposes only. Not a formal legal opinion. Final patent risk assessment requires attorney review."

**dependency_unavailable 处理**：wrapper 未实现 → MCP 返回 dependency_unavailable → 该次 tool_call_record.run_status 标记，但 step 不 crash。Step 13 → `review_status` = `partial`/`failed`；Step 14 → `patent_review_status` = `partial`/`completed_with_warnings`/`failed`。

**Graph 拓扑**：当前 MVP 在 `build_step1_14_graph` 中按 Step 12 → 13 → 14 → END **顺序**执行。架构 v0.1 目标是 Step 13 ∥ Step 14 在 Step 12 之后并行；两个 agent 独立、artifact 互不依赖，未来切换到 LangGraph 并行分支为机械改动（见 `app/graph/adc_graph.py` 顶部 docstring）。

Step 13/14 API 同样走 `execution_decision` 门控：`wait_for_input` / `blocked` → 409，graph 节点 → skipped + `executed=False`。

## Step 10-12 — External scoring handoff、validation、deterministic ranking

Step 10/11/12 都是**确定性 service**（不调 LLM，不调 MCP）。Step 10 是**外部 Yufei AEE 的 handoff 准备**——不是本地评分。

| Step | 服务 | artifact | 关键状态 |
|---|---|---|---|
| 10 | `ScoringHandoffService.prepare(run_id)` | `scoring_handoff_package.json` | `awaiting_external_scoring` / `partial` / `failed` |
| 11 | `ScoringValidationService.validate(run_id)` | `scoring_validation.json` | `awaiting_external_input` / `completed` / `completed_with_warnings` / `failed` |
| 12 | `RankingService.build_ranking_table(run_id)` | `ranking_table.json` | `awaiting_external_scoring` / `completed` / `failed` / `skipped` |

**外部 scoring 结果输入约定**：
- 文件位置：`{run_id}/inputs/external_scoring_result.json`
- shape：
  ```json
  {
    "scored_at": "2026-06-15T12:00:00Z",
    "candidates": [
      {
        "candidate_id": "candidate_xxxx",
        "total_score": 7.4,
        "dimensions": {
          "docking_score": -8.2,
          "developability_score": 6.5,
          "evidence_score": 7.0,
          "patent_risk_score": 2.1
        },
        "notes": "optional"
      }
    ]
  }
  ```
- Step 11 校验：`candidate_id` 必须存在；`total_score` 必须数字且默认 `[0, 10]`（超出区间记 warning）；`dimensions.*` 数字校验（`docking_score` 允许 `[-50, 50]`）；未知 `candidate_id` 记 warning。
- raw 行**不内嵌**到 Step 11 artifact —— 只保留 `external_scoring_input_ref` 指向 `inputs/external_scoring_result.json`，per-row 问题进 `issues[]`。

**没有外部 scoring 时**：
- Step 10 输出 `handoff_status="awaiting_external_scoring"` + `expected_result_storage_path` 写明把结果文件该放哪里。
- Step 11 输出 `validation_status="awaiting_external_input"`，`validated_candidate_ids=[]`，`row_count=0`，notes 提示如何 unblock。
- Step 12 输出 `ranking_status="awaiting_external_scoring"`，`ranked_candidates=[]`。
- **从不**伪造排名。`build_step1_12_graph` 会按 Step 10→11→12 跑完 12 个 step；Step 11/12 以 awaiting 状态收尾，graph 不 crash。

Step 12 排序：当 Step 11 完成且 `validated_candidate_ids` 非空，按 `total_score` 降序、ties 按 `candidate_id` 升序，确保 deterministic、可重复。

Step 10/11/12 API 完全沿用 `execution_decision` 门控（`wait_for_input` / `blocked` → 409）。

## Step 7-9 — StructureAndDesignAgent

`StructureAndDesignAgent`（`app/agents/structure_and_design_agent.py`）合并管理 Step 7/8/9 的主链路。
所有工具调用走 agent 自带的 inventory-scoped `MCPClient`，从不绕过 `scope_filter`。

| Step | 入口方法 | 主要 MCP tools (v0.2 inventory) | 输出 artifact |
|---|---|---|---|
| 7 | `run_step_7(run_id)` | `RCSBData_get_entry` (per pdb_id enrichment) | `prepared_structure_input_package.json` |
| 8 | `run_step_8(run_id)` | `CrystalStructure_validate` / `RCSBData_get_entry` / `get_refinement_resolution_by_pdb_id` / `ProteinsPlus_profile_structure_quality` / `alphafold_get_prediction` | `structure_prediction_and_interface_results.json` |
| 9 | `run_step_9(run_id)` | `ZINC_search_by_smiles` / `ZINC_get_compound` / `ZINC_search_compounds` (architecture carve-out — see `AGENT_TOOL_OVERRIDES` in `app/mcp/scope_filter.py`) | `compound_screening_artifact.json` |

Raw tool outputs：每次 MCP 调用的 raw payload 写到 `tool_outputs/step_{07|08|09}/{tool_call_id}.json`，
normalized artifact 里只放 `tool_call_records[].tool_output_ref` / `.tool_output_artifact_id`，
以及 Step 8 的 `output_artifacts[].storage_ref`。
**raw payload 从不嵌入 `prepared_structure_inputs / candidate_structure_results / compound_hits`**。

ZINC 版本处理（架构 v0.1 + Week 3 audit）：当前 ToolUniverse 的 `ZINC_*` wrapper 实际访问的是 ZINC15 endpoint，
但 mock 模式下我们无法证实任何 ZINC22 来源。Step 9 的 `CompoundHit` 一律使用：
- `source_library = "ZINC"`（家族级，不写 ZINC22）
- `source_database_version = "unknown"`（除非上游显式确认）
- `source_tool_name = <调用的工具名>`
- `source_runtime_status` 跟随 MCP `run_status`

任何 wrapper 抛 `NotImplementedError`（捕获后转为 `dependency_unavailable`），step 仍然完成并标 `partial`，**不会让整个 Step 7/8/9 崩掉**。

Step 7/8/9 API 完全沿用 Step 5/6 的 `execution_decision` 门控：plan_status 为 `wait_for_input` / `blocked` → 409，graph 节点 → workflow_state.skipped + `results.step_XX.executed=False`。

## Step 4 — Deterministic policy engine 与 plan_status 全局门控

`WorkflowSetupService` (`app/services/workflow_setup_service.py`) 是 deterministic
policy engine：读取 Step 2 `structured_query` 与 Step 3 `input_readiness_status`，
产出 `run_step_plan`，包含两层语义：

- **全局门控**：`plan_status ∈ {ready_to_execute | wait_for_input | blocked}`
- **逐 step 决策**：`planned_steps[].planned_status ∈ {run | partial | skip | blocked | wait_for_input}`，
  附 `reason` 与 `lane_flags`

Step 5/6 的执行入口（LangGraph 节点 + HTTP API）共用一个 helper
`execution_decision(plan, step_id)` 来判断是否放行，门控顺序：

1. `plan_status == ready_to_execute` 才允许继续；否则 **agent 不会被调用**。
2. 再看本 step 的 `planned_status`；`skip`/`blocked` 也会拦截。

| `plan_status` | LangGraph 节点行为 | HTTP API 行为 |
|---|---|---|
| `ready_to_execute` | 检查 per-step → 正常 run / skip | 200 + 正常 artifact |
| `wait_for_input` | 不调用 agent，标记 `workflow_state.step_05/06 = skipped`，`results.step_XX = {executed: False, plan_status, reason}` | **409** `WorkflowStateError`，detail 含 `{plan_status, planned_status, step_id, reason}` |
| `blocked` | 同上 | 同上 |

`wait_for_input` 表示用户需要补输入（payload 缺失等 warning 级 gap）；
`blocked` 表示存在 blocking gap（缺 target）或 Step 4 policy 拒绝继续。两种情况下
**不会写出 Step 5/6 正常 artifact，registry 中对应 id 保持 None**，避免被误认为执行过。

`workflow_state` 没有 `blocked` 这个状态值；按用户要求统一使用 `skipped` + `results` 中
的 `reason / plan_status` 字段表达，schema 不动。

## 测试

```bash
pytest tests/schemas/        # IO schema round-trip
pytest tests/services/       # 确定性 service
pytest tests/mcp/            # FastMCP 注册集合 ⊆ v0.2 inventory
pytest tests/e2e/            # Step 1→4 本地链路
```

## 硬约束

- 不要手写 A2A 协议，统一走 `python-a2a`
- 不要手写 MCP 协议，统一走 `python-a2a` FastMCP
- MCP 工具来源仅 `项目文件/ToolUniversity_inventory_v0.2.xlsx`，禁止注册 ToolUniverse 全量
- 每个 step 的 I/O 严格按 `ADC_Pipeline_IO_Schema_v0.1.md`，不发明新 schema
- 原始 MCP 输出走 S3 引用，不嵌入业务 artifact
- 当前 step API 文件 **一 step 一文件**，不要合并 router
