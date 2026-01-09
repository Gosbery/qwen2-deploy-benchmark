# Phase 4 — Serving & Concurrency Benchmark Log

**Project:** qwen2-deploy-benchmark  
**Model:** Qwen/Qwen2-1.5B-Instruct  
**Hardware:** RTX 2060 (6GB)  
**Backend:** PyTorch + Transformers  
**Serving:** FastAPI + Uvicorn  
**Precision:** FP16  

---

## Phase 4 Goal

验证在 **单 GPU、资源受限环境** 下：

- 模型是否可以稳定服务化
- 并发请求对延迟、吞吐、稳定性的影响
- 是否存在请求排队（queue wait）
- 如何以工程方式正确测试并发

---

## Experiment 4.1 — FP16 基础服务化

**内容**
- 使用 FastAPI + Uvicorn 暴露 `/generate`
- 单并发（`asyncio.Semaphore(1)`）
- 固定参数：`ctx_len=2048`, `max_new_tokens=64`

**结果**
- 服务可稳定启动
- 单请求延迟、显存占用与 Phase 2 baseline 一致
- GPU 峰值显存 ~2.9 GB

**结论**
- FP16 模型在 RTX 2060 上可稳定服务化
- 服务化本身未引入额外显存压力

---

## Experiment 4.2 — 手动并发测试（失败方法）

**内容**
- 使用 `/generate_stream`
- 打开多个终端，人工同时发送请求
- 观察 `queue_wait_s`

**观察**
- 多次实验中 `queue_wait_s ≈ 0`
- 行为不稳定，难以复现

**问题定位**
- 人工操作无法保证请求真正同时到达
- Streaming 会提前返回控制权
- Python / OS 调度噪声大于人为时间差

**结论**
- **手动并发测试在工程上不可用**
- 方法本身不成立，而非实现问题

---

## Experiment 4.3 — Streaming Endpoint 的局限性

**内容**
- 深入分析 `/generate_stream` 行为
- 对比 streaming 与 non-streaming

**发现**
- Streaming endpoint 会掩盖真实排队
- 适合用户体验，不适合容量/并发评估

**结论**
- Streaming ≠ Benchmark
- 并发测试必须使用 non-streaming 接口

---

## Experiment 4.4 — 引入脚本化并发测试

**内容**
- 编写 `bench_generate.py`
- 使用 `/generate`（non-stream）
- 参数：
  - concurrency = 5
  - total_requests = 15
  - ctx_len = 2048
  - max_new_tokens = 64

**目的**
- 强制制造真实并发
- 消除人工操作不确定性
- 获取统计意义上的结果

---

## Experiment 4.5 — 并发基准测试（第一次）

**结果**
- 成功率：93.33%（14/15）
- 平均延迟：~17.6 s
- P99 延迟：~30 s
- tokens/sec：~8.6
- 出现 1 次 `ConnectionResetError`

**分析**
- GPU 完全饱和
- 请求发生隐式排队
- 系统进入不稳定区间

---

## Experiment 4.6 — 并发基准测试（复测）

**结果**
- 成功率：100%（15/15）
- 平均延迟：~17.7 s
- P99 延迟：~23.5 s
- tokens/sec：~8.6

**结论**
- 并发显著拉高延迟
- 排队行为在统计层面明确存在
- 单 GPU 服务存在明显容量上限

---

## Phase 4 Final Conclusions

1. 并发问题必须通过 **脚本化 benchmark** 测试
2. 手动并发测试不可复现，不具工程价值
3. Streaming endpoint 不适合做容量评估
4. 单 GPU 下，并发会显著放大尾延迟
5. Phase 4 验证了“可服务”与“可扩展”的边界

---

## Phase 4 Status

**Status: Complete**

- 服务化完成
- 并发测试方法正确
- 结果可复现
- 工程结论明确

---

## Next Phase

**Phase 3 — Quantization (INT8 / INT4)**

原因：
- Phase 4 已确认 GPU 是主要瓶颈
- Quantization 是唯一现实的纵向优化手段
- Phase 4 的 benchmark 工具可直接复用
