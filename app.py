"""
LlamaIndex RAG Web UI（基于 Streamlit）

LlamaIndex 官方推荐搭配（streamlit.io/partners/llamaindex）。
复用 run_vector_demo.py 的所有函数，不重写业务逻辑。

功能：
  - Tab1 问答：聊天界面，三种检索模式（向量/聚合/全文）
  - Tab2 Ingest：文件上传 + 目录导入 + 实时进度
  - Tab3 知识库管理：列表、状态、节点数、存储大小

启动：
    streamlit run app.py
    或
    .venv\\Scripts\\streamlit run app.py
"""
import os
import sys
import time
import shutil
from pathlib import Path

import streamlit as st
import torch
import yaml

# ---- 复用 run_vector_demo.py 的所有函数 ----
# run_vector_demo.py 有 if __name__ == "__main__" 保护，导入安全
from run_vector_demo import (
    # 知识库管理
    _load_kb_configs,
    _load_or_build_kb,
    _select_single_kb,  # 不直接用（CLI 交互），但参考其逻辑
    # 索引/存储
    make_pipeline,
    get_storage_size,
    format_size,
    # 转载去重（SimHash）
    _get_simhash_store,
    _filter_near_duplicates,
    # 检索模式
    RETRIEVAL_MODES,
    _is_aggregate_query,
    _handle_aggregate_query,
    _handle_fulltext_query,
    _get_fulltext_searcher,
    FullTextSearcher,
    _make_query_engine,
    # 展示
    _highlight_relevant_sentences,
    # 常量
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INDEX_ID,
    _RETR_CFG,
)

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# =====================================================================
# 页面配置
# =====================================================================
st.set_page_config(
    page_title="LlamaIndex RAG 管理面板",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# 会话状态初始化
# =====================================================================
def init_session_state():
    """初始化 Streamlit 会话状态。"""
    defaults = {
        "llm": None,
        "embed_model": None,
        "index": None,
        "query_engine": None,
        "current_kb_id": None,
        "current_kb_cfg": None,
        "retrieval_mode": "vector",
        "models_loaded": False,
        "chat_history": [],
        "fulltext_searcher": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =====================================================================
# 模型加载（用 session_state 缓存，避免 @cache_resource 热重载冲突）
# =====================================================================
def get_llm():
    """获取 DeepSeek LLM（session_state 缓存）。"""
    if "llm_instance" not in st.session_state:
        st.session_state.llm_instance = DeepSeek(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            temperature=0.3,
            max_tokens=2048,
        )
    return st.session_state.llm_instance


def get_embed_model():
    """加载 bge-m3 embedding 模型（session_state 缓存）。"""
    if "embed_instance" not in st.session_state:
        st.session_state.embed_instance = HuggingFaceEmbedding(
            model_name=r"C:\code\LlamaIndex\models\bge-m3",
            embed_batch_size=64,
            model_kwargs={"torch_dtype": torch.float16},
        )
    return st.session_state.embed_instance


def init_global_settings():
    """设置 LlamaIndex 全局 Settings。"""
    if not st.session_state.models_loaded:
        with st.spinner("加载 LLM 和 Embedding 模型（首次约 30 秒）..."):
            Settings.llm = get_llm()
            Settings.embed_model = get_embed_model()
            Settings.chunk_size = CHUNK_SIZE
            st.session_state.models_loaded = True


def load_index_with_progress(storage_dir: str, kb_name: str = ""):
    """加载已有索引或创建空索引，显示预估进度条。

    有索引 → 加载（显示进度条）
    无索引 → 创建空索引（快速，显示提示）
    """
    import threading

    index_exists = os.path.exists(os.path.join(storage_dir, "docstore.json"))

    if not index_exists:
        # 空库：创建空索引
        st.info(f"📭 创建空数据库: {kb_name}")
        from llama_index.core import VectorStoreIndex
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex(storage_context=storage_context)
        index.set_index_id(INDEX_ID)
        os.makedirs(storage_dir, exist_ok=True)
        index.storage_context.persist(storage_dir)
        st.success(f"✅ 空数据库已创建 | 存储到: {storage_dir}")
        st.info("💡 请切换到「📥 Ingest」Tab 添加文档")
        return index

    storage_size = get_storage_size(storage_dir)
    # 预估速率 20 MB/s（JSON 反序列化 + 对象构建综合速率，偏保守）
    estimated_time = max(storage_size / (20 * 1024 * 1024), 2.0)

    st.info(f"存储容量: {format_size(storage_size)} | 预估加载 {estimated_time:.0f}s")

    progress_bar = st.progress(0, text=f"加载中（预估 {estimated_time:.0f}s）...")
    status_text = st.empty()

    stop_event = threading.Event()

    def _push_progress():
        t0 = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - t0
            if elapsed <= estimated_time:
                pct = elapsed / estimated_time * 90
            else:
                # 超过预估时间，缓慢推进到 95%
                pct = min(90 + (elapsed - estimated_time) * 2, 95)
            pct_int = int(pct)
            progress_bar.progress(pct_int, text=f"加载进度 {pct_int}% | {elapsed:.1f}s")
            time.sleep(0.2)

    pt = threading.Thread(target=_push_progress, daemon=True)
    pt.start()

    t0_load = time.time()
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context, index_id=INDEX_ID)
    finally:
        stop_event.set()
        pt.join()

    elapsed = time.time() - t0_load
    progress_bar.progress(100, text=f"✅ 加载完成 | 耗时 {elapsed:.1f}s")
    return index


# =====================================================================
# 侧边栏：知识库选择 + 检索模式
# =====================================================================
def render_sidebar():
    """渲染侧边栏。"""
    st.sidebar.title("📰 RAG 管理面板")
    st.sidebar.markdown("---")

    # ---- 知识库选择 ----
    st.sidebar.subheader("📋 知识库")
    configs = _load_kb_configs()
    kb_options = {f"{cfg.get('name', kb_id)} [{kb_id}]": kb_id for kb_id, cfg in configs.items()}
    kb_labels = list(kb_options.keys())

    selected_label = st.sidebar.selectbox(
        "选择知识库",
        options=kb_labels,
        index=0,
        help="切换知识库后需点击「加载」按钮",
    )
    selected_kb_id = kb_options[selected_label]
    cfg = configs[selected_kb_id]
    storage_dir = cfg.get("storage_dir", f"./storage/{selected_kb_id}")
    index_exists = os.path.exists(os.path.join(storage_dir, "docstore.json"))

    # 显示该数据库状态
    if index_exists:
        st.sidebar.caption(f"✅ 已有数据 | 大小: {format_size(get_storage_size(storage_dir))}")
    else:
        st.sidebar.caption(f"📭 空数据库（加载后可用 ingest 添加）")

    # 加载按钮
    if st.sidebar.button("🔄 加载/切换数据库", type="primary"):
        if selected_kb_id == st.session_state.current_kb_id:
            st.sidebar.info("已加载当前数据库")
        else:
            # 无论有无索引都可以加载
            try:
                index = load_index_with_progress(storage_dir, cfg.get("name", selected_kb_id))
                st.session_state.index = index
                st.session_state.query_engine = _make_query_engine(index)
                st.session_state.current_kb_id = selected_kb_id
                st.session_state.current_kb_cfg = cfg
                st.session_state.fulltext_searcher = None
                st.sidebar.success(f"✅ 已加载: {cfg.get('name', selected_kb_id)}")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ 加载失败: {e}")

    # 显示当前知识库状态
    if st.session_state.current_kb_id:
        cfg = st.session_state.current_kb_cfg
        st.sidebar.markdown(f"""
        **当前知识库**
        - ID: `{st.session_state.current_kb_id}`
        - 名称: {cfg.get('name', '')}
        - 存储: `{cfg.get('storage_dir', '')}`
        """)
        storage_size = get_storage_size(cfg.get("storage_dir", ""))
        st.sidebar.markdown(f"- 大小: {format_size(storage_size)}")
    else:
        st.sidebar.warning("⚠️ 未加载知识库")

    st.sidebar.markdown("---")

    # ---- 检索模式选择 ----
    st.sidebar.subheader("🔍 检索模式")
    mode_keys = list(RETRIEVAL_MODES.keys())
    mode_labels = [f"{k} - {RETRIEVAL_MODES[k]}" for k in mode_keys]
    current_mode_idx = mode_keys.index(st.session_state.retrieval_mode) if st.session_state.retrieval_mode in mode_keys else 0

    selected_mode_label = st.sidebar.radio(
        "选择检索模式",
        options=mode_labels,
        index=current_mode_idx,
        help="向量=语义匹配(LLM回答) | 聚合=正则直遍(100%覆盖) | 全文=BM25排序(关键词匹配)",
    )
    new_mode = mode_keys[mode_labels.index(selected_mode_label)]
    if new_mode != st.session_state.retrieval_mode:
        st.session_state.retrieval_mode = new_mode
        st.rerun()

    # 全文搜索索引状态
    if st.session_state.retrieval_mode == "fulltext" and st.session_state.index:
        searcher = _get_fulltext_searcher(st.session_state.index)
        st.sidebar.markdown(f"""
        **全文搜索索引**
        - 状态: {searcher.status}
        """)
        if not searcher._built:
            if st.sidebar.button("🔨 构建倒排索引"):
                with st.spinner("构建倒排索引 ..."):
                    searcher.build()
                st.rerun()

    st.sidebar.markdown("---")

    # ---- 模型信息 ----
    st.sidebar.subheader("🤖 模型")
    st.sidebar.markdown("""
    - **LLM**: DeepSeek
    - **Embedding**: bge-m3
    - **Chunk**: 512 / 50 overlap
    """)
    if st.session_state.models_loaded:
        st.sidebar.success("✅ 模型已加载")
    else:
        st.sidebar.warning("⏳ 模型未加载")

    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 *基于 Streamlit + LlamaIndex*")


# =====================================================================
# Tab1: 问答
# =====================================================================
def render_chat_tab():
    """渲染问答聊天 Tab。"""
    st.header("💬 智能问答")

    if not st.session_state.index:
        # ---- 未加载知识库：主区域显示一键加载 ----
        st.markdown("### 🚀 一键加载知识库")

        configs = _load_kb_configs()
        kb_options = {f"{cfg.get('name', kb_id)} [{kb_id}]": kb_id for kb_id, cfg in configs.items()}
        kb_labels = list(kb_options.keys())

        # 默认选当前侧边栏选中的那个
        default_idx = 0
        if st.session_state.current_kb_id and st.session_state.current_kb_id in kb_options.values():
            default_idx = list(kb_options.values()).index(st.session_state.current_kb_id)

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_label = st.selectbox("选择知识库", options=kb_labels, index=default_idx, key="main_kb_select")
        with col2:
            st.markdown("#### ")  # 对齐高度
            load_clicked = st.button("🚀 一键加载", type="primary", use_container_width=True)

        selected_kb_id = kb_options[selected_label]
        cfg = configs[selected_kb_id]
        storage_dir = cfg.get("storage_dir", f"./storage/{selected_kb_id}")
        index_exists = os.path.exists(os.path.join(storage_dir, "docstore.json"))

        # 显示选中数据库状态
        if index_exists:
            storage_size = get_storage_size(storage_dir)
            st.success(f"✅ 已有数据 | 大小: {format_size(storage_size)} | 存储路径: `{storage_dir}`")
        else:
            st.info(f"📭 空数据库 | 加载后可用 ingest 添加文档 | 存储路径: `{storage_dir}`")

        # 点击加载（无论有无索引都可以加载）
        if load_clicked:
            if selected_kb_id == st.session_state.current_kb_id:
                st.info("该数据库已加载")
            else:
                try:
                    index = load_index_with_progress(storage_dir, cfg.get("name", selected_kb_id))
                    st.session_state.index = index
                    st.session_state.query_engine = _make_query_engine(index)
                    st.session_state.current_kb_id = selected_kb_id
                    st.session_state.current_kb_cfg = cfg
                    st.session_state.fulltext_searcher = None
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 加载失败: {e}")

        # 帮助信息
        st.markdown("---")
        st.info("""
        **使用说明**：
        1. 上方下拉框选择知识库（仅显示 ✅ 已有索引的可加载）
        2. 点击「🚀 一键加载」按钮
        3. 等待进度条加载完成
        4. 加载完成后自动进入问答界面

        **提示**：未构建索引的知识库需先在 CLI 中运行 `python run_vector_demo.py` 构建索引。
        """)
        return

    mode = st.session_state.retrieval_mode
    st.info(f"当前检索模式: **{RETRIEVAL_MODES[mode]}**")

    # 聊天历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander(f"📄 检索片段 ({len(msg['sources'])} 个)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**片段 {i}** | 来源: `{src['file']}`")
                        st.text(src["content"][:500])
                        st.markdown("---")

    # 输入框
    if question := st.chat_input("输入你的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # 生成回答
        with st.chat_message("assistant"):
            try:
                sources = []

                if mode == "aggregate":
                    # 聚合直遍模式
                    with st.spinner("聚合直遍 docstore ..."):
                        # 重定向 stdout 捕获输出
                        from io import StringIO
                        old_stdout = sys.stdout
                        sys.stdout = captured = StringIO()
                        _handle_aggregate_query(question, st.session_state.index)
                        sys.stdout = old_stdout
                        answer = captured.getvalue()
                        # 简化输出
                        answer = answer if answer else "（聚合查询完成，详见上方输出）"

                elif mode == "fulltext":
                    # 全文搜索模式
                    with st.spinner("BM25 全文搜索 ..."):
                        searcher = _get_fulltext_searcher(st.session_state.index)
                        if not searcher._built:
                            with st.spinner("首次使用，构建倒排索引 ..."):
                                searcher.build()
                        results = searcher.search(question, top_k=_RETR_CFG["fulltext"]["top_k"])
                        answer_parts = [f"**全文搜索结果：共 {len(results)} 条（BM25 排序）**\n"]
                        for i, (doc_id, score, node) in enumerate(results[:10], 1):
                            meta = node.metadata or {}
                            file_path = meta.get("file_path", "未知")
                            content = node.get_content()[:200]
                            answer_parts.append(f"**{i}.** score={score:.4f} | `{file_path}`\n> {content}...\n")
                        answer = "\n".join(answer_parts)
                        for doc_id, score, node in results[:5]:
                            sources.append({
                                "file": (node.metadata or {}).get("file_path", "未知"),
                                "content": node.get_content(),
                            })

                else:
                    # 向量检索模式（默认）
                    with st.spinner("向量检索 + LLM 生成回答 ..."):
                        # 聚合查询检测
                        if _is_aggregate_query(question):
                            st.warning("⚠️ 检测到聚合查询，向量检索模式仅返回 top-k 片段。建议切换到「聚合」模式获取全部结果。")

                        response = st.session_state.query_engine.query(question)
                        answer = str(response)

                        # 收集 source nodes
                        if getattr(response, "source_nodes", None):
                            for node in response.source_nodes:
                                meta = node.node.metadata or {}
                                sources.append({
                                    "file": meta.get("file_path", "未知"),
                                    "content": node.node.get_content(),
                                })

                st.markdown(answer)
                if sources:
                    with st.expander(f"📄 检索片段 ({len(sources)} 个)"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**片段 {i}** | 来源: `{src['file']}`")
                            st.text(src["content"][:500])
                            st.markdown("---")

                # 保存到历史
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources if sources else None,
                })

            except Exception as e:
                st.error(f"❌ 查询出错: {type(e).__name__}: {e}")

    # 清除聊天历史按钮
    if st.session_state.chat_history:
        if st.button("🗑️ 清除聊天历史"):
            st.session_state.chat_history = []
            st.rerun()


# =====================================================================
# Tab2: Ingest
# =====================================================================
def render_ingest_tab():
    """渲染 Ingest 管理 Tab。"""
    st.header("📥 Ingest 数据导入")

    if not st.session_state.current_kb_id:
        st.warning("⚠️ 请先在侧边栏加载知识库")
        return

    cfg = st.session_state.current_kb_cfg
    st.info(f"当前知识库: **{cfg.get('name', '')}** | 存储: `{cfg.get('storage_dir', '')}`")

    # ---- 方式1：从目录批量导入 ----
    st.subheader("📁 从目录导入")
    default_data_dir = cfg.get("data_dir", "")
    data_dir = st.text_input("数据源目录", value=default_data_dir, key="ingest_dir")

    if st.button("🚀 开始 Ingest", type="primary"):
        if not os.path.isdir(data_dir):
            st.error(f"❌ 目录不存在: {data_dir}")
        else:
            _do_ingest(data_dir)

    st.markdown("---")

    # ---- 方式2：文件上传 ----
    st.subheader("📤 上传文件")
    uploaded_files = st.file_uploader(
        "选择 .md 文件上传",
        type=["md"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files and st.button("📥 导入上传的文件"):
        _do_ingest_uploaded(uploaded_files)

    st.markdown("---")

    # ---- Ingest 状态显示 ----
    if "ingest_status" in st.session_state:
        st.subheader("📊 最近 Ingest 状态")
        status = st.session_state.ingest_status
        col1, col2, col3 = st.columns(3)
        col1.metric("文档数", status.get("total_docs", "-"))
        col2.metric("节点数", status.get("total_nodes", "-"))
        col3.metric("耗时", f"{status.get('elapsed', 0):.1f}s")

        if status.get("logs"):
            with st.expander("详细日志"):
                for log in status["logs"]:
                    st.text(log)


def _do_ingest(data_dir: str):
    """执行从目录的 ingest。"""
    if not st.session_state.index:
        st.error("❌ 索引未加载")
        return

    progress_bar = st.progress(0, text="读取文档中 ...")
    status_text = st.empty()
    logs = []

    try:
        t0 = time.time()

        # 1) 读取文档
        logs.append(f"[1/4] 读取目录: {data_dir}")
        status_text.text("[1/4] 读取文档 ...")
        reader = SimpleDirectoryReader(
            input_dir=data_dir,
            required_exts=[".md"],
            recursive=True,
        )
        documents = reader.load_data()
        logs.append(f"  -> 读取到 {len(documents)} 个文档")
        progress_bar.progress(25, text=f"读取到 {len(documents)} 个文档")

        # 2) 构建 pipeline
        logs.append("[2/4] 构建去重管道 ...")
        status_text.text("[2/4] 构建 IngestionPipeline ...")
        storage_context = StorageContext.from_defaults(
            persist_dir=st.session_state.current_kb_cfg["storage_dir"]
        )
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
                Settings.embed_model,
            ],
            docstore_strategy="duplicates_only",
            docstore=storage_context.docstore,
        )
        progress_bar.progress(50, text="向量化中 ...")

        # 2.5) SimHash 转载过滤（与 CLI 共用同一套指纹库，近似转载直接跳过）
        simhash_store = _get_simhash_store(
            st.session_state.current_kb_cfg["storage_dir"],
            docstore=storage_context.docstore,
        )
        near_dups = []
        if simhash_store is not None:
            documents, near_dups = _filter_near_duplicates(documents, simhash_store)
            logs.append(f"  -> SimHash 转载过滤：拦截 {len(near_dups)} 篇，保留 {len(documents)} 篇")

        # 3) 运行 pipeline
        logs.append("[3/4] 运行 pipeline（切块 + 去重 + 向量化）...")
        status_text.text("[3/4] 切块 + 去重 + 向量化 ...")
        nodes = pipeline.run(documents=documents, show_progress=False)
        logs.append(f"  -> 生成 {len(nodes)} 个节点")
        progress_bar.progress(75, text=f"生成 {len(nodes)} 个节点，插入索引 ...")

        # 4) 插入 + 持久化
        logs.append("[4/4] 插入索引 + 持久化 ...")
        status_text.text("[4/4] 插入索引 + 保存 ...")
        st.session_state.index.insert_nodes(nodes)
        storage_context.persist(persist_dir=st.session_state.current_kb_cfg["storage_dir"])
        if simhash_store is not None:
            simhash_store.save()

        # 刷新 query_engine
        st.session_state.query_engine = _make_query_engine(st.session_state.index)

        elapsed = time.time() - t0
        progress_bar.progress(100, text=f"完成！{len(documents)} 文档 → {len(nodes)} 节点，耗时 {elapsed:.1f}s")

        st.session_state.ingest_status = {
            "total_docs": len(documents),
            "total_nodes": len(nodes),
            "elapsed": elapsed,
            "logs": logs,
        }

        st.success(f"✅ Ingest 完成！处理 {len(documents)} 文档（拦截转载 {len(near_dups)} 篇），生成 {len(nodes)} 节点，耗时 {elapsed:.1f}s")

    except Exception as e:
        progress_bar.progress(100, text="失败")
        st.error(f"❌ Ingest 失败: {type(e).__name__}: {e}")
        st.exception(e)


def _do_ingest_uploaded(uploaded_files: list):
    """执行上传文件的 ingest。"""
    if not st.session_state.index:
        st.error("❌ 索引未加载")
        return

    # 保存上传文件到临时目录
    temp_dir = Path("./_temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    for f in uploaded_files:
        (temp_dir / f.name).write_bytes(f.getvalue())

    st.info(f"已保存 {len(uploaded_files)} 个文件到 {temp_dir}，开始 ingest ...")
    _do_ingest(str(temp_dir))

    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# Tab3: 知识库管理
# =====================================================================
def render_kb_management_tab():
    """渲染知识库管理 Tab。"""
    st.header("🗂️ 知识库管理")

    configs = _load_kb_configs()

    # 统计
    total = len(configs)
    has_index = sum(1 for kb_id, cfg in configs.items() if os.path.exists(os.path.join(cfg.get("storage_dir", ""), "docstore.json")))
    col1, col2, col3 = st.columns(3)
    col1.metric("知识库总数", total)
    col2.metric("已构建索引", has_index)
    col3.metric("未构建", total - has_index)

    st.markdown("---")

    # 知识库列表
    st.subheader("📋 知识库列表")
    for kb_id, cfg in configs.items():
        storage_dir = cfg.get("storage_dir", f"./storage/{kb_id}")
        index_exists = os.path.exists(os.path.join(storage_dir, "docstore.json"))
        size = get_storage_size(storage_dir) if index_exists else 0

        with st.expander(f"{'✅' if index_exists else '⬜'} {cfg.get('name', kb_id)} [{kb_id}]"):
            col1, col2 = st.columns(2)
            col1.markdown(f"""
            **配置信息**
            - ID: `{kb_id}`
            - 名称: {cfg.get('name', '')}
            - 数据源: `{cfg.get('data_dir', '')}`
            - 存储目录: `{storage_dir}`
            - 索引状态: {'✅ 已构建' if index_exists else '⬜ 未构建'}
            """)
            col2.markdown(f"""
            **存储信息**
            - 大小: {format_size(size) if index_exists else '-'}
            - 路径: `{storage_dir}`
            """)

            if index_exists and kb_id == st.session_state.current_kb_id:
                # 显示当前知识库的详细统计
                try:
                    docstore = st.session_state.index.storage_context.docstore
                    node_count = len(docstore.docs)
                    st.markdown(f"**节点数**: {node_count}")
                except Exception:
                    pass

            # 操作按钮
            col_btn1, col_btn2 = st.columns(2)
            if col_btn1.button("🔄 加载", key=f"load_{kb_id}"):
                try:
                    index = load_index_with_progress(storage_dir, cfg.get("name", kb_id))
                    st.session_state.index = index
                    st.session_state.query_engine = _make_query_engine(index)
                    st.session_state.current_kb_id = kb_id
                    st.session_state.current_kb_cfg = cfg
                    st.session_state.fulltext_searcher = None
                    st.success(f"✅ 已加载: {cfg.get('name', kb_id)}")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 加载失败: {e}")

            if col_btn2.button("🗑️ 删除索引", key=f"del_{kb_id}"):
                if os.path.exists(storage_dir):
                    shutil.rmtree(storage_dir, ignore_errors=True)
                    st.success(f"已删除索引: {storage_dir}")
                    st.rerun()


# =====================================================================
# 主函数
# =====================================================================
def main():
    init_session_state()

    # 初始化模型（全局，只加载一次）
    init_global_settings()

    # 侧边栏
    render_sidebar()

    # 主区域 Tabs
    tab1, tab2, tab3 = st.tabs(["💬 问答", "📥 Ingest", "🗂️ 知识库管理"])

    with tab1:
        render_chat_tab()
    with tab2:
        render_ingest_tab()
    with tab3:
        render_kb_management_tab()


if __name__ == "__main__":
    main()
