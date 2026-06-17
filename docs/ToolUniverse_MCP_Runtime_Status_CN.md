# ToolUniverse MCP Runtime 当前状态说明

更新时间：2026-06-17  
用途：用于汇报 ADC Agent 后端中 MCP / ToolUniverse 工具真实运行接入状态。本文是中文摘要版；详细审计记录见项目文件中的 `ToolUniverse_Runtime_Integration_Audit_v0.1.md`。

## 1. 当前架构

后端没有把 ToolUniverse 全量工具注册进 MCP。MCP server 只注册 `ToolUniversity_inventory_v0.2.xlsx` 中进入项目范围的工具，当前 smoke 测试确认 MCP runtime 注册工具数为 81。

ToolUniverse runtime 只在一个地方被直接导入：

```python
app/mcp/tooluniverse_adapter.py
```

所有已经迁移的 wrapper 都通过统一路径调用 ToolUniverse：

```text
Agent
  -> LocalMCPClient
  -> MCP wrapper
  -> ToolUniverseAdapter
  -> ToolUniverse.run_one_function(...)
```

这个设计避免每个 wrapper 自己 import `ToolUniverse`，也避免重复初始化、绕过 inventory scope、或产生多套外部 API 调用逻辑。

## 2. Live / Mock 控制方式

默认情况下，wrapper 走 `_live=False`，返回 deterministic mock envelope，用于本地测试和稳定回归。

真实 ToolUniverse 调用由两个环境变量控制：

```bash
MCP_LIVE_TOOLS=true
MCP_LIVE_TOOL_ALLOWLIST=EuropePMC_search_articles,SwissADME_calculate_adme
```

当 `MCP_LIVE_TOOLS=true` 且 tool name 命中 allowlist 时，`LocalMCPClient` 会向 wrapper 注入 `_live=True`。wrapper 随后调用 `tooluniverse_adapter.call_tool(...)`。

如果 allowlist 未命中，或 `MCP_LIVE_TOOLS=false`，工具仍走 mock 路径。这样可以逐个工具开启 live，不会一次性触发所有外部服务。

ToolUniverse 中的 agentic LLM 工具需要 Gemini key 时，adapter 会把项目 Settings 中的 `gemini_api_key` / `gemini_model` 安全桥接到 `os.environ`，供 ToolUniverse 内部读取。已有系统环境变量优先，不会被覆盖。key 不会写入 artifact，也不会打印。

## 3. 当前 Live-Ready Step 状态

### Step 5 Candidate Context

状态：所有非 ZINC 工具已经 ToolUniverseAdapter-backed。

已完成范围包括：

- ChEMBL molecule / drug / similarity / substructure 工具
- SAbDab 工具
- TheraSAbDab 工具
- IEDB BCR sequence 工具

ZINC 5 个工具仍然 `intentionally_disabled`：

- `ZINC_search_compounds`
- `ZINC_get_compound`
- `ZINC_search_by_smiles`
- `ZINC_search_by_properties`
- `ZINC_get_purchasable`

原因：当前 ZINC upstream captcha-gated / unstable。系统不会把 ZINC 标成 `live_ready`，也不会默认标成 ZINC22。

### Step 6 Developability

状态：大部分可迁移工具已经 ToolUniverseAdapter-backed。

已完成范围包括：

- ChEMBL developability family，包含 binding sites close-out
- DrugProps：QED / Lipinski / PAINS
- BindingDB target lookup
- PROSITE sequence scan
- EBIProteins features / epitopes / antigen
- GlyGen glycoprotein / glycosylation site
- iPTMnet PTM sites
- IEDB MHC-I binding prediction
- PDBePISA interfaces / monomer analysis
- PDBe-KB interface residues
- SwissADME ADME / drug-likeness

仍 deferred：

- ADMETAI family
- ProteinsPlus structure quality / binding site tools

### Step 7 Structure Retrieval

状态：6/6 完成 ToolUniverseAdapter-backed。

已完成：

- `alphafold_get_prediction`
- `RCSBData_get_entry`
- `RCSBData_get_assembly`
- `PDBeSearch_search_structures`
- `SAbDab_get_structure`
- `RCSBAdvSearch_search_structures`

### Step 8 Structure Prediction / Validation

状态：可迁移工具已经完成。

已完成：

- `get_refinement_resolution_by_pdb_id`
- `CrystalStructure_validate`
- `PDBePISA_get_interfaces`（与 Step 6 共享 binding）

仍 deferred：

- NvidiaNIM AlphaFold2 multimer / OpenFold3 / Boltz2

原因：需要 Nvidia GPU / vendor key。

### Step 9 Structural Variant / Compound Screening

状态：AlphaMissense 已完成，其余高风险工具 deferred。

已完成：

- `AlphaMissense_get_variant_score`

仍 deferred：

- `DynaMut2_predict_stability`
- `ESM_generate_protein_sequence`
- `ESM_score_variant_sae_batch`
- NvidiaNIM ProteinMPNN / RFdiffusion

### Step 13 Scientific Evidence

状态：8/8 完成 ToolUniverseAdapter-backed。

已完成：

- `EuropePMC_search_articles`
- `openalex_search_works`
- `PubTator3_LiteratureSearch`
- `PubTator3_get_annotations`
- `SemanticScholar_search_papers`
- `ChEMBL_search_documents`
- `LiteratureSearchTool`
- `MultiAgentLiteratureSearch`

`MultiAgentLiteratureSearch` 是 ToolUniverse ComposeTool，会使用内部 LLM agents。为控制成本和运行时间，wrapper 将 `max_iterations` hard-clamp 到 1。

### Step 14 Patent / Prior Art

状态：主要公开 patent 工具已完成。

已完成：

- `PubChem_get_associated_patents_by_CID`
- `FDA_OrangeBook_get_patent_info`

仍 deferred：

- `drugbank_get_drug_references_by_drug_name_or_id`

原因：DrugBank 需要 license / key。

## 4. Deferred 工具和原因

| 工具类别 | 当前状态 | 原因 |
|---|---|---|
| ZINC ×5 | intentionally_disabled | upstream captcha-gated / unstable；不能声明 ZINC22 |
| ADMETAI ×8 | deferred | ToolUniverse `ADMETAITool` 需要 `torch` + `admet_ai` + heavy model weights |
| ProteinsPlus ×3 | deferred | ToolUniverse schema 是 async REST jobs，`max_wait_time` 900/1800 秒；binding-site 工具还有 raw `pdb_content` 文件语义 |
| DynaMut2 | deferred | async REST job，最长轮询约 300 秒 |
| ESM ×2 | deferred | 需要 vendor `ESM_API_KEY`、额外 `esm` 依赖和远程推理计费 |
| NvidiaNIM ×5 | vendor_gpu | 需要 Nvidia GPU / vendor key |
| DrugBank | key_required | 需要 DrugBank license / key |

Deferred 工具不会静默返回 success。测试已确认这些工具即使被加入 live allowlist，也不会调用 fake ToolUniverseAdapter，而是以 `dependency_unavailable` / `executor="deferred"` 暴露。

## 5. Runtime Readiness Audit 结果

已新增 runtime readiness audit，覆盖真实调用路径：

```text
Agent -> LocalMCPClient -> wrapper -> ToolUniverseAdapter
```

覆盖的 agent path：

| Step | Agent / module | 代表工具 |
|---|---|---|
| Step 5 | CandidateContextAgent | `ChEMBL_search_molecules`, `SAbDab_search_structures`, `TheraSAbDab_search_by_target` |
| Step 6 | DevelopabilityAgent | `ChEMBL_search_activities`, `SwissADME_calculate_adme` |
| Step 9 | Structure / Design agent | `AlphaMissense_get_variant_score` |
| Step 13 | EvidenceAgent | `EuropePMC_search_articles`, `MultiAgentLiteratureSearch` |
| Step 14 | PatentIPAgent | `PubChem_get_associated_patents_by_CID`, `FDA_OrangeBook_get_patent_info` |

验证结论：

- `MCP_LIVE_TOOLS=true` 且 allowlist 命中时，返回 `executor="tooluniverse"`。
- allowlist 未命中或 `MCP_LIVE_TOOLS=false` 时，返回 `executor="mock"`。
- Deferred 工具返回 `executor="deferred"`，不会静默 success。
- 跨 step 调用会被 scope filter 拒绝，返回 `tool_not_in_agent_scope`。
- ZINC 即使被放入 allowlist，也不会进入 live adapter。
- raw ToolUniverse payload 只保留在 adapter envelope / 后续 `tool_output_ref` 指向的 raw artifact 中，不进入 normalized artifact。

为方便审计，`LocalMCPClient.call_tool` 现在会在结果顶层标记 `executor`：

| Outcome | run_status | executor |
|---|---|---|
| ToolUniverse live path | `success` | `tooluniverse` |
| Mock path | `success` | `mock` |
| Deferred / NotImplementedError | `dependency_unavailable` | `deferred` |
| Cross-step scope refusal | `skipped` | `unknown` |
| Unexpected error | `failed` | `error` |

这个字段只用于 runtime/audit 可读性，没有改动 ToolCallRecord schema 或 FastMCP/A2A 协议。

## 6. 当前测试结果

最近确认结果：

```text
targeted MCP / readiness suite: 275 passed, 1 warning
full pytest: 520 passed, 1 skipped
run_mcp_smoke.py: PASS
MCP registered tools: 81
```

`run_step1_14_graph.py` 在真实 Gemini provider 下可能受到外部 Gemini 503 / quota / malformed JSON 影响。该问题属于外部 LLM 稳定性，不是 MCP ToolUniverse migration failure。使用 `LLM_PROVIDER=mock` 可以做 deterministic regression。

## 7. 本地启用 live mode 示例

只开启少量工具：

```bash
export MCP_LIVE_TOOLS=true
export MCP_LIVE_TOOL_ALLOWLIST=EuropePMC_search_articles,SwissADME_calculate_adme,FDA_OrangeBook_get_patent_info
```

如果需要 ToolUniverse 内部 agentic LLM 工具，例如 `MultiAgentLiteratureSearch`：

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=...
export GEMINI_MODEL=gemini-3.5-flash
```

本地 `.env` 已被 `.gitignore` 忽略，不要提交到 GitHub。运行 smoke 或测试时也不要打印 key、完整 prompt 或完整 ToolUniverse payload。

## 8. 当前结论

MCP ToolUniverse migration 的核心路径已经可用：

- MCP registration 仍然受 v0.2 inventory 限制。
- ToolUniverse runtime 通过统一 adapter 调用。
- 已迁移工具可以在 agent path 中被 live allowlist 安全触发。
- Deferred 工具有明确原因并 fail safely。
- raw payload isolation 已通过测试保护。

下一步重点不应继续硬迁移高风险工具，而应进入真实业务样例测试：选择少量 allowlisted live tools，跑受控 Step 1-14 或局部 Step smoke，观察真实 upstream 稳定性、quota、latency 和 artifact 质量。
