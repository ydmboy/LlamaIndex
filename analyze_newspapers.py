"""
分析所有报纸的支持情况，判断支持不佳的原因（网站问题 vs 软件问题）。

分析维度：
  1. 数据源完整性：文件是否存在、是否为空、文件大小
  2. 文件格式：是否 .md、是否有 ### 标题、是否有正文
  3. 解析能力：能否被 SimpleDirectoryReader 读取
  4. 切片效果：切块后节点数、平均长度
  5. 检索能力：用固定查询测试向量检索是否返回结果

输出：newspaper_analysis_report.md
"""
import os
import sys
import time
import json
import torch
from pathlib import Path
from collections import defaultdict

# 复用主项目模块
from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

WIKI_ROOT = Path("D:/wiki")
EMBED_MODEL_PATH = r"C:\code\LlamaIndex\models\bge-m3"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
OUTPUT_REPORT = Path("newspaper_analysis_report.md")

# 固定测试查询（通用，对所有报纸公平）
TEST_QUERIES = ["经济发展", "科技创新", "民生政策", "党的建设", "文化教育"]


def analyze_all():
    """分析所有报纸，返回分析结果列表。"""
    results = []

    # 1) 加载 embedding 模型（只加载一次）
    print("[1/5] 加载 embedding 模型 ...", flush=True)
    t0 = time.time()
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_PATH,
        embed_batch_size=64,
        model_kwargs={"torch_dtype": torch.float16},
    )
    Settings.chunk_size = CHUNK_SIZE
    print(f"    -> 完成，耗时 {time.time()-t0:.1f}s", flush=True)

    # 2) 扫描所有报纸目录
    newspaper_dirs = sorted([d for d in WIKI_ROOT.iterdir() if d.is_dir()])
    print(f"[2/5] 发现 {len(newspaper_dirs)} 份报纸，开始分析 ...", flush=True)

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for i, np_dir in enumerate(newspaper_dirs, 1):
        np_name = np_dir.name
        print(f"\n[{i}/{len(newspaper_dirs)}] 分析: {np_name}", flush=True)
        result = analyze_one_newspaper(np_dir, splitter)
        results.append(result)
        # 实时输出简要结论
        status = result["status"]
        reason = result.get("reason", "")
        print(f"    -> {status} | {reason}", flush=True)

    # 3) 生成报告
    print(f"\n[3/5] 生成分析报告 ...", flush=True)
    generate_report(results)
    print(f"    -> 报告已保存到 {OUTPUT_REPORT}", flush=True)

    # 4) 输出汇总
    print("\n" + "=" * 60)
    print("分析汇总")
    print("=" * 60)
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["status"]] += 1
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} 份")
    print("=" * 60)


def analyze_one_newspaper(np_dir: Path, splitter: SentenceSplitter) -> dict:
    """分析单份报纸。返回分析结果字典。"""
    result = {
        "name": np_dir.name,
        "path": str(np_dir),
        "status": "未知",
        "reason": "",
        "details": {},
    }

    # ---- 维度1：数据源完整性 ----
    # 查找日期子目录
    date_dirs = [d for d in np_dir.iterdir() if d.is_dir()]
    if not date_dirs:
        result["status"] = "不支持"
        result["reason"] = "网站问题：无日期子目录，数据源缺失"
        result["details"]["error"] = "no_date_subdir"
        return result

    # 取最新日期目录
    date_dir = sorted(date_dirs)[-1]
    result["details"]["date_dir"] = date_dir.name

    # 查找 .md 文件
    md_files = list(date_dir.glob("*.md"))
    result["details"]["md_count"] = len(md_files)

    if not md_files:
        result["status"] = "不支持"
        result["reason"] = "网站问题：日期目录下无 .md 文件"
        result["details"]["error"] = "no_md_files"
        return result

    # 统计文件大小
    total_size = sum(f.stat().st_size for f in md_files)
    result["details"]["total_size_kb"] = round(total_size / 1024, 1)

    if total_size < 1000:  # 小于 1KB
        result["status"] = "支持不佳"
        result["reason"] = "网站问题：文件过小（<1KB），可能抓取失败或内容为空"
        result["details"]["error"] = "file_too_small"
        return result

    # ---- 维度2：文件格式检查 ----
    # 读取第一个文件检查内容结构
    sample_file = md_files[0]
    try:
        sample_content = sample_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        result["status"] = "支持不佳"
        result["reason"] = f"软件问题：文件读取失败 ({type(e).__name__}: {e})"
        result["details"]["error"] = "read_error"
        return result

    # 检查是否有 ### 标题（标准格式）
    heading_count = sample_content.count("\n### ")
    result["details"]["sample_headings"] = heading_count
    result["details"]["sample_length"] = len(sample_content)

    # 检查是否有明显的乱码（非中文非ASCII字符比例）
    chinese_chars = sum(1 for c in sample_content if '\u4e00' <= c <= '\u9fff')
    ascii_chars = sum(1 for c in sample_content if c.isascii() and c.isprintable())
    total_chars = len(sample_content)
    if total_chars > 0:
        readable_ratio = (chinese_chars + ascii_chars) / total_chars
    else:
        readable_ratio = 0
    result["details"]["readable_ratio"] = round(readable_ratio, 2)

    if readable_ratio < 0.5:
        result["status"] = "支持不佳"
        result["reason"] = "网站问题：内容乱码率高（可读字符 <50%），可能编码问题"
        result["details"]["error"] = "encoding_issue"
        return result

    if heading_count == 0 and len(sample_content) < 500:
        result["status"] = "支持不佳"
        result["reason"] = "网站问题：无标准标题格式且内容过短，可能非报纸原文"
        result["details"]["error"] = "no_structure"
        return result

    # ---- 维度3：解析能力 ----
    try:
        reader = SimpleDirectoryReader(
            input_dir=str(date_dir),
            required_exts=[".md"],
            recursive=True,
            exclude=[".obsidian"],
            filename_as_id=True,
        )
        documents = reader.load_data()
        result["details"]["doc_count"] = len(documents)
        if not documents:
            result["status"] = "支持不佳"
            result["reason"] = "软件问题：SimpleDirectoryReader 未读取到任何文档"
            result["details"]["error"] = "reader_empty"
            return result
    except Exception as e:
        result["status"] = "支持不佳"
        result["reason"] = f"软件问题：解析失败 ({type(e).__name__}: {e})"
        result["details"]["error"] = "parse_error"
        return result

    # ---- 维度4：切片效果 ----
    try:
        nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
        result["details"]["node_count"] = len(nodes)
        if nodes:
            avg_len = sum(len(n.get_content()) for n in nodes) / len(nodes)
            result["details"]["avg_node_length"] = round(avg_len, 0)
        else:
            result["status"] = "支持不佳"
            result["reason"] = "软件问题：切片后无节点（SentenceSplitter 失败）"
            result["details"]["error"] = "split_empty"
            return result
    except Exception as e:
        result["status"] = "支持不佳"
        result["reason"] = f"软件问题：切片失败 ({type(e).__name__}: {e})"
        result["details"]["error"] = "split_error"
        return result

    # ---- 维度5：检索能力（轻量测试，不构建完整索引）----
    # 用固定查询测试 embedding + 余弦相似度
    try:
        # 对前 50 个节点做 embedding（避免全量太慢）
        test_nodes = nodes[:50] if len(nodes) > 50 else nodes
        node_texts = [n.get_content() for n in test_nodes]
        node_embeddings = Settings.embed_model.get_text_embedding_batch(node_texts)

        # 对测试查询做 embedding
        query_embeddings = Settings.embed_model.get_text_embedding_batch(TEST_QUERIES)

        # 计算每个查询的最大相似度
        import numpy as np
        node_emb_array = np.array(node_embeddings)
        max_similarities = []
        for q_emb in query_embeddings:
            q_array = np.array(q_emb)
            sims = node_emb_array @ q_array / (
                np.linalg.norm(node_emb_array, axis=1) * np.linalg.norm(q_array) + 1e-8
            )
            max_similarities.append(float(sims.max()))

        avg_max_sim = sum(max_similarities) / len(max_similarities)
        result["details"]["avg_max_similarity"] = round(avg_max_sim, 3)
        result["details"]["test_queries"] = TEST_QUERIES
        result["details"]["max_similarities"] = [round(s, 3) for s in max_similarities]

        if avg_max_sim < 0.3:
            result["status"] = "支持不佳"
            result["reason"] = f"软件问题：检索相似度过低（avg={avg_max_sim:.3f}），embedding 匹配差"
            result["details"]["error"] = "low_similarity"
            return result

    except Exception as e:
        result["status"] = "支持不佳"
        result["reason"] = f"软件问题：embedding 失败 ({type(e).__name__}: {e})"
        result["details"]["error"] = "embed_error"
        return result

    # ---- 综合判定 ----
    if len(nodes) < 10:
        result["status"] = "支持一般"
        result["reason"] = f"内容偏少（{len(nodes)} 节点），检索覆盖有限"
    elif avg_max_sim < 0.5:
        result["status"] = "支持一般"
        result["reason"] = f"检索相似度中等（avg={avg_max_sim:.3f}），部分查询可能不准"
    else:
        result["status"] = "支持良好"
        result["reason"] = f"数据完整（{len(documents)} 文档/{len(nodes)} 节点），检索相似度高（avg={avg_max_sim:.3f}）"

    return result


def generate_report(results: list):
    """生成 Markdown 分析报告。"""
    lines = [
        "# 报纸支持情况分析报告",
        "",
        f"**分析时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**报纸总数**：{len(results)}",
        f"**Embedding 模型**：bge-m3",
        f"**测试查询**：{TEST_QUERIES}",
        "",
        "## 1. 汇总统计",
        "",
    ]

    # 汇总统计
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["status"]] += 1

    lines.append("| 状态 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for status in ["支持良好", "支持一般", "支持不佳", "不支持"]:
        count = status_counts.get(status, 0)
        pct = count / len(results) * 100 if results else 0
        lines.append(f"| {status} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## 2. 问题分类统计",
        "",
    ])

    # 按原因分类
    reason_counts = defaultdict(int)
    for r in results:
        if r["status"] in ["支持不佳", "不支持"]:
            # 提取问题类型（网站问题/软件问题）
            reason = r.get("reason", "")
            if "网站问题" in reason:
                reason_counts["网站问题"] += 1
            elif "软件问题" in reason:
                reason_counts["软件问题"] += 1
            else:
                reason_counts["其他问题"] += 1

    lines.append("| 问题类型 | 数量 |")
    lines.append("|----------|------|")
    for rtype, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {rtype} | {count} |")

    lines.extend([
        "",
        "## 3. 详细分析",
        "",
        "| 报纸 | 状态 | 原因 | 文档数 | 节点数 | 平均长度 | 相似度 |",
        "|------|------|------|--------|--------|----------|--------|",
    ])

    # 按状态排序：不支持 > 支持不佳 > 支持一般 > 支持良好
    status_order = {"不支持": 0, "支持不佳": 1, "支持一般": 2, "支持良好": 3}
    sorted_results = sorted(results, key=lambda r: (status_order.get(r["status"], 99), r["name"]))

    for r in sorted_results:
        d = r["details"]
        doc_count = d.get("doc_count", "-")
        node_count = d.get("node_count", "-")
        avg_len = d.get("avg_node_length", "-")
        sim = d.get("avg_max_similarity", "-")
        lines.append(f"| {r['name']} | {r['status']} | {r['reason']} | {doc_count} | {node_count} | {avg_len} | {sim} |")

    lines.extend([
        "",
        "## 4. 支持不佳/不支持的详细原因",
        "",
    ])

    for r in sorted_results:
        if r["status"] in ["支持不佳", "不支持"]:
            lines.extend([
                f"### {r['name']}",
                f"- **状态**：{r['status']}",
                f"- **原因**：{r['reason']}",
                f"- **路径**：{r['path']}",
                f"- **详情**：{json.dumps(r['details'], ensure_ascii=False)}",
                "",
            ])

    lines.extend([
        "## 5. 原因分析说明",
        "",
        "### 网站问题（数据源本身）",
        "- **无日期子目录 / 无 .md 文件**：抓取脚本未成功下载内容",
        "- **文件过小（<1KB）**：抓取失败或页面内容为空",
        "- **乱码率高（可读字符 <50%）**：编码问题或抓取到非正文内容",
        "- **无标准标题格式**：可能抓取到非报纸原文页面",
        "",
        "### 软件问题（LlamaIndex 处理）",
        "- **SimpleDirectoryReader 未读取到文档**：文件格式不被识别",
        "- **切片后无节点**：SentenceSplitter 无法处理内容",
        "- **embedding 失败**：模型无法向量化文本",
        "- **检索相似度过低**：embedding 匹配差，可能需要更换模型",
        "",
        "### 建议",
        "1. **网站问题**：检查抓取脚本，确认目标网站结构是否变化",
        "2. **软件问题**：检查 LlamaIndex 版本、embedding 模型兼容性",
        "3. **支持一般**：可尝试调大 similarity_top_k 或更换 embedding 模型",
        "",
    ])

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print("=" * 60)
    print("报纸支持情况分析工具")
    print("=" * 60)
    print(f"数据源: {WIKI_ROOT}")
    print(f"Embedding: {EMBED_MODEL_PATH}")
    print(f"测试查询: {TEST_QUERIES}")
    print("=" * 60)
    analyze_all()
    print("\n分析完成！")
