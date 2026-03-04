"""
CoT生成器 V2 主程序
支持真实工具调用和5类推理场景
"""
import argparse
import logging
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from api_client import APIClient
from dataset_loader import DatasetLoader
from real_tool_executor import RealToolExecutor
from image_processor import ImageProcessor
from cot_generator_v3 import CoTGeneratorV2
from reasoning_classifier import ReasoningTypeClassifier
from ssh_image_fetcher import SSHImageFetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cot_generator_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CoT生成器 V2 - 真实工具集成版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 基础用法
  python main_v2.py --dataset data.jsonl --output results.json

  # 只生成Type B（纠错场景）
  python main_v2.py --dataset data.jsonl --output results.json --type-filter B

  # 使用特定模型和并发数
  python main_v2.py --dataset data.jsonl --output results.json \\
    --model claude-3.5-sonnet --max-workers 5

  # 启用缓存并指定工具路径
  python main_v2.py --dataset data.jsonl --output results.json \\
    --enable-cache --tools-interface-path ../agents/fundus_tools_agent
"""
    )

    # 必需参数
    parser.add_argument("--dataset", required=True, help="VQA数据集路径（JSON或JSONL格式）")
    parser.add_argument("--output", required=True, help="输出文件路径")

    # 模型配置
    parser.add_argument("--model", default=None, help="模型名称（默认从.env读取）")
    parser.add_argument("--api-key", default=None, help="API密钥（默认从.env读取）")
    parser.add_argument("--base-url", default=None, help="API基础URL（默认从.env读取）")

    # 工具配置
    parser.add_argument(
        "--tools-interface-path",
        default=os.getenv("TOOLS_INTERFACE_PATH", "../agents/fundus_tools_agent"),
        help="tools_interface.py所在目录（默认: 从.env读取TOOLS_INTERFACE_PATH，未设置则为../agents/fundus_tools_agent）"
    )
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        default=os.getenv("ENABLE_CACHE", "false").lower() == "true",
        help="启用工具结果缓存（默认: 从.env读取ENABLE_CACHE）"
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("CACHE_DIR", "./tool_cache"),
        help="缓存目录（默认: 从.env读取CACHE_DIR，未设置则为./tool_cache）"
    )

    # 生成配置
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("MAX_WORKERS", "3")),
        help="最大并发数（默认: 从.env读取MAX_WORKERS，未设置则为3）"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="保存间隔（每N个样本保存一次，默认: 5）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="批处理大小（可选，默认处理全部）"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="起始索引（默认: 0）"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        help="结束索引（可选）"
    )

    # 过滤器
    parser.add_argument(
        "--type-filter",
        choices=["A", "B", "C", "D", "E"],
        help="只生成特定类型的轨迹（A=标准SOP, B=纠错, C=级联, D=多病, E=拒诊）"
    )
    parser.add_argument(
        "--skip-healthy",
        action="store_true",
        help="跳过A-healthy类型的API调用，节省token（工具仍会执行以检测动态升级为Type B的可能）"
    )

    # 其他选项
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式（只分类，不生成轨迹）"
    )
    parser.add_argument(
        "--image-root",
        help="图片根目录（用于解析相对路径）"
    )
    parser.add_argument(
        "--no-ssh",
        action="store_true",
        help="禁用SSH，使用本地路径模式（适用于在服务器上直接运行）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="启用断点续跑（从上次中断处继续生成）"
    )
    parser.add_argument(
        "--clear-progress",
        action="store_true",
        help="清除断点续跑进度文件，从头开始"
    )

    args = parser.parse_args()

    # 加载环境变量
    load_dotenv()

    logger.info("=" * 60)
    logger.info("CoT生成器 V2 - 真实工具集成版本")
    logger.info("=" * 60)

    # 1. 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    loader = DatasetLoader()
    examples = loader.load_dataset(args.dataset, image_root=args.image_root)

    # 切片
    if args.end_idx:
        examples = examples[args.start_idx:args.end_idx]
    elif args.start_idx > 0:
        examples = examples[args.start_idx:]

    if args.batch_size:
        examples = examples[:args.batch_size]

    logger.info(f"共加载 {len(examples)} 个样本")

    # 2. 类型过滤和统计
    if args.type_filter or args.dry_run:
        logger.info("分析推理类型分布...")
        classifier = ReasoningTypeClassifier()

        type_stats = defaultdict(int)
        filtered_examples = []
        unclassified_count = 0

        for ex in examples:
            rtype = classifier.classify(ex.question, ex.ground_truth_answer)

            # 检查是否成功分类
            if rtype is None:
                unclassified_count += 1
                logger.debug(f"样本无法分类: {ex.image_path}")
                # 无法分类的样本不加入过滤列表（跳过）
                continue

            type_stats[f"{rtype.type}-{rtype.subtype}"] += 1

            if args.type_filter:
                if args.type_filter == "B":
                    combined_text = ReasoningTypeClassifier._extract_answer_from_question(
                        ex.question, ex.ground_truth_answer
                    )
                    gt_lower = combined_text.lower()
                    might_become_b = (
                        any(kw in gt_lower for kw in ["diabetic retinopathy"]) or
                        any(kw in gt_lower for kw in ["amd","drusen", "age-related macular", "age related macular", "macular degeneration"]) or
                        any(kw in gt_lower for kw in ["healthy", "normal", "physiologic", "生理性", "正常"])
                    )
                    if might_become_b:
                        filtered_examples.append(ex)
                elif args.type_filter == "E":
                    filtered_examples.append(ex)
                elif rtype.type == args.type_filter:
                    filtered_examples.append(ex)
            else:
                filtered_examples.append(ex)

        # 记录无法分类的样本数量
        if unclassified_count > 0:
            logger.info(f"⚠️ {unclassified_count} 个样本无法分类，已跳过")

        # 打印统计
        logger.info("\n推理类型分布:")
        for rtype, count in sorted(type_stats.items()):
            logger.info(f"  {rtype}: {count}")

        if args.type_filter:
            examples = filtered_examples
            if args.type_filter == "B":
                logger.info(f"\n类型过滤（Type B动态过滤）:")
                logger.info(f"  - 保留GT=AMD的样本（可能误判为DR）")
                logger.info(f"  - 保留GT=Healthy的样本（可能误判为Glaucoma/病灶）")
                logger.info(f"  - 保留GT=Healthy的样本（看到DR病灶但可能是伪影）")
                logger.info(f"类型过滤后剩余 {len(examples)} 个样本")
            elif args.type_filter == "E":
                logger.info(f"\n类型过滤（Type E动态过滤）:")
                logger.info(f"  - 保留所有样本（任何图像都可能因质量差而成为Type E）")
                logger.info(f"  - 将对所有样本执行质量评估")
                logger.info(f"类型过滤后剩余 {len(examples)} 个样本")
            else:
                logger.info(f"\n类型过滤后剩余 {len(examples)} 个样本")

        if args.dry_run:
            logger.info("\n试运行模式完成（未生成轨迹）")
            return

    # 3. 初始化组件
    logger.info("\n初始化组件...")

    # API客户端
    try:
        api_client = APIClient(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url
        )
        logger.info(f"✓ API客户端初始化成功: {api_client.model}")
    except Exception as e:
        logger.error(f"✗ API客户端初始化失败: {e}")
        return

    # 工具执行器
    try:
        tool_executor = RealToolExecutor(
            tools_interface_path=args.tools_interface_path,
            enable_cache=args.enable_cache,
            cache_dir=args.cache_dir
        )
        logger.info(f"✓ 工具执行器初始化成功")
        if args.enable_cache:
            stats = tool_executor.get_cache_stats()
            logger.info(f"  缓存: {stats['total_entries']} 条记录")
    except Exception as e:
        logger.error(f"✗ 工具执行器初始化失败: {e}")
        logger.warning("将使用模拟模式继续")
        tool_executor = RealToolExecutor(
            tools_interface_path=None,
            enable_cache=False
        )

    # 图像处理器
    image_processor = ImageProcessor()
    logger.info("✓ 图像处理器初始化成功")

    # SSH图片获取器
    ssh_fetcher = SSHImageFetcher(
        ssh_host=os.getenv("SSH_HOST"),
        ssh_user=os.getenv("SSH_USER"),
        ssh_password=os.getenv("SSH_PASSWORD"),  # 支持密码认证
        ssh_port=int(os.getenv("SSH_PORT", "22")),
        remote_base_path=os.getenv("REMOTE_BASE_PATH", "<ANON_ABS_PATH>"),
        local_cache_dir=os.getenv("LOCAL_CACHE_DIR", "./image_cache"),
        ssh_key_path=os.getenv("SSH_KEY_PATH"),
        use_scp=os.getenv("USE_SCP", "true").lower() == "true"
    )
    if ssh_fetcher.enabled:
        auth_info = {
            "key": "密钥认证",
            "password": "密码认证",
            "agent": "ssh-agent"
        }.get(ssh_fetcher.auth_method, "未知")

        logger.info(f"✓ SSH图片获取器初始化成功: {ssh_fetcher.ssh_user}@{ssh_fetcher.ssh_host} ({auth_info})")

        # 测试连接
        if not ssh_fetcher.test_connection():
            logger.warning("⚠️  SSH连接测试失败，将使用本地路径模式")
    else:
        logger.info("✓ SSH未配置，将使用本地路径模式")

    # CoT生成器
    allowed_types = [args.type_filter] if args.type_filter else None
    generator = CoTGeneratorV2(
        api_client=api_client,
        tool_executor=tool_executor,
        image_processor=image_processor,
        ssh_fetcher=ssh_fetcher,
        max_workers=args.max_workers,
        allowed_types=allowed_types,  # 传递类型过滤器
        disable_ssh=args.no_ssh,  # 传递SSH禁用标志
        skip_healthy=args.skip_healthy  # 传递跳过healthy标志
    )
    if allowed_types:
        logger.info(f"✓ CoT生成器初始化成功 (并发数: {args.max_workers}, 类型过滤: {allowed_types})")
    else:
        logger.info(f"✓ CoT生成器初始化成功 (并发数: {args.max_workers})")

    if args.skip_healthy:
        logger.info("✓ 已启用 skip_healthy: A-healthy 类型将跳过API调用")


    # 4. 批量生成
    logger.info("\n" + "=" * 60)
    logger.info("开始批量生成轨迹")
    logger.info("=" * 60 + "\n")

    # 断点续跑：检查进度文件
    progress_file = Path(args.output).parent / f"{Path(args.output).stem}_progress.json"
    completed_indices = set()
    existing_results = []

    if args.clear_progress:
        if progress_file.exists():
            progress_file.unlink()
            logger.info(f"✓ 已清除进度文件: {progress_file}")

    # 始终传递 progress_file 路径（无论是否 resume 都要保存进度）
    progress_file_path = str(progress_file)

    if args.resume and progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                completed_indices = set(progress_data.get('completed_indices', []))

            if completed_indices:
                logger.info(f"✓ 发现进度文件: {progress_file}")
                logger.info(f"  已完成 {len(completed_indices)} 个样本")
                logger.info(f"  剩余 {len(examples) - len(completed_indices)} 个样本待生成")

            # 加载已生成的结果
            if Path(args.output).exists():
                try:
                    with open(args.output, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                    logger.info(f"  已加载 {len(existing_results)} 个已生成结果")
                except Exception as e:
                    logger.warning(f"  无法加载已存在的结果文件: {e}")
                    existing_results = []
            else:
                logger.info(f"  结果文件不存在，将创建新文件")
        except Exception as e:
            logger.warning(f"读取进度文件失败: {e}，将从头开始")
            completed_indices = set()
            existing_results = []
    elif args.resume:
        logger.info(f"⚠️  未找到进度文件 ({progress_file_path})，将从头开始生成")
    else:
        logger.info(f"💾 进度将保存到: {progress_file_path}")

    results = generator.batch_generate_v2(
        examples=examples,
        output_path=args.output,
        save_interval=args.save_interval,
        completed_indices=completed_indices,
        existing_results=existing_results,
        progress_file=progress_file_path  # 始终传递进度文件路径
    )

    # 5. 统计报告
    logger.info("\n" + "=" * 60)
    logger.info("生成完成！统计报告:")
    logger.info("=" * 60)

    total = len(results)
    success = sum(1 for r in results if r.get("trajectory") is not None)
    failed = total - success

    logger.info(f"总样本数: {total}")
    logger.info(f"成功生成: {success} ({success/total*100:.1f}%)")
    logger.info(f"生成失败: {failed} ({failed/total*100:.1f}%)")

    # 类型分布
    type_distribution = defaultdict(int)
    for r in results:
        if r.get("metadata"):
            rtype = f"{r['metadata']['reasoning_type']}-{r['metadata']['reasoning_subtype']}"
            type_distribution[rtype] += 1

    if type_distribution:
        logger.info("\n推理类型分布:")
        for rtype, count in sorted(type_distribution.items()):
            logger.info(f"  {rtype}: {count} ({count/total*100:.1f}%)")

    # 工具使用统计
    tool_usage = defaultdict(int)
    for r in results:
        if r.get("tool_execution_log"):
            for log in r["tool_execution_log"]:
                tool_usage[log["tool_name"]] += 1

    if tool_usage:
        logger.info("\n工具使用统计:")
        for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {tool}: {count}次")

    # 保存统计报告
    report_path = args.output.replace(".json", "_report.json")
    report = {
        "total": total,
        "success": success,
        "failed": failed,
        "type_distribution": dict(type_distribution),
        "tool_usage": dict(tool_usage),
        "config": vars(args)
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n统计报告已保存: {report_path}")
    logger.info(f"轨迹结果已保存: {args.output}")
    logger.info("\n" + "=" * 60)
    logger.info("全部完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n用户中断，退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n发生错误: {e}", exc_info=True)
        sys.exit(1)
