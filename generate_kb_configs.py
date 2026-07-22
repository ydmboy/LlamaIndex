"""
扫描 D:\\wiki 下所有报纸目录，自动生成 kb_configs/*.yaml 配置文件。
让已有数据全部可在系统中选择使用。

文件名格式：北京日报-2026-06-30-全刊.md
中文名提取：按 "-" 分割取第一段
"""
import os
from pathlib import Path
import yaml

WIKI_ROOT = Path("D:/wiki")
KB_CONFIGS_DIR = Path("c:/code/LlamaIndex/kb_configs")
KB_CONFIGS_DIR.mkdir(exist_ok=True)


def extract_chinese_name(md_file: Path) -> str:
    """从 .md 文件名提取报纸中文名。
    格式：北京日报-2026-06-30-全刊.md → 北京日报
    """
    stem = md_file.stem  # 北京日报-2026-06-30-全刊
    parts = stem.split("-")
    if parts:
        return parts[0]
    return stem


def main():
    print(f"扫描 {WIKI_ROOT} ...")
    newspaper_dirs = sorted([d for d in WIKI_ROOT.iterdir() if d.is_dir()])
    print(f"发现 {len(newspaper_dirs)} 份报纸目录\n")

    created = 0
    skipped = 0
    results = []

    for np_dir in newspaper_dirs:
        np_id = np_dir.name
        yaml_path = KB_CONFIGS_DIR / f"{np_id}.yaml"

        # 查找日期子目录
        date_dirs = sorted([d for d in np_dir.iterdir() if d.is_dir()])
        if not date_dirs:
            print(f"[跳过] {np_id}: 无日期子目录")
            skipped += 1
            results.append((np_id, "跳过", "无日期子目录"))
            continue

        # 取最新日期
        date_dir = date_dirs[-1]
        date_str = date_dir.name

        # 查找 .md 文件
        md_files = list(date_dir.glob("*.md"))
        if not md_files:
            print(f"[跳过] {np_id}: 无 .md 文件")
            skipped += 1
            results.append((np_id, "跳过", "无 .md 文件"))
            continue

        # 提取中文名
        chinese_name = extract_chinese_name(md_files[0])

        # 构建配置
        config = {
            "name": chinese_name,
            "description": f"{chinese_name}全文数据",
            "data_dir": f"D:\\\\wiki\\\\{np_id}\\\\{date_str}",
            "file_exts": [".md"],
        }

        # 如果配置已存在，跳过（不覆盖用户手动编辑的）
        if yaml_path.exists():
            print(f"[已存在] {np_id} ({chinese_name})")
            skipped += 1
            results.append((np_id, "已存在", chinese_name))
            continue

        # 写入配置文件
        # 手动构建 yaml 内容，保持与 beijing_daily.yaml 一致的格式
        yaml_content = f"""# {chinese_name}知识库
name: "{chinese_name}"
description: "{chinese_name}全文数据"
data_dir: "D:\\\\wiki\\\\{np_id}\\\\{date_str}"
file_exts:
  - ".md"
"""
        yaml_path.write_text(yaml_content, encoding="utf-8")
        print(f"[创建] {np_id} -> {chinese_name} (数据: {date_str})")
        created += 1
        results.append((np_id, "创建", chinese_name))

    # 汇总
    print(f"\n{'='*60}")
    print(f"批量生成完成")
    print(f"{'='*60}")
    print(f"  新建配置: {created} 份")
    print(f"  已存在/跳过: {skipped} 份")
    print(f"  配置目录: {KB_CONFIGS_DIR}")
    print(f"{'='*60}")

    # 列出所有配置文件
    all_configs = sorted(KB_CONFIGS_DIR.glob("*.yaml"))
    all_configs = [c for c in all_configs if not c.name.endswith(".example")]
    print(f"\n当前可用知识库配置（{len(all_configs)} 个）：")
    for cfg in all_configs:
        with open(cfg, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        name = data.get("name", "?")
        print(f"  {cfg.stem:30s} -> {name}")


if __name__ == "__main__":
    main()
