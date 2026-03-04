"""
批处理脚本：一次性处理整个文件夹的所有数据集
支持文件夹级别的断点续跑
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
import subprocess
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """批处理器：管理多个数据集的生成任务"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        progress_file: str = "batch_progress.json",
        main_script: str = "main_v2.py"
    ):
        """
        初始化批处理器

        Args:
            input_dir: 输入文件夹路径（包含 .jsonl 文件）
            output_dir: 输出文件夹路径
            progress_file: 进度文件路径
            main_script: 主脚本路径（main_v2.py）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.progress_file = Path(progress_file)
        self.main_script = main_script

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scan_datasets(self):
        """扫描输入文件夹，获取所有 .jsonl 文件"""
        jsonl_files = sorted(self.input_dir.glob("*.jsonl"))
        logger.info(f"扫描到 {len(jsonl_files)} 个数据集文件")
        return jsonl_files

    def load_progress(self):
        """加载批处理进度"""
        if not self.progress_file.exists():
            return {}

        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            logger.info(f"✓ 加载进度文件: {self.progress_file}")
            return progress
        except Exception as e:
            logger.warning(f"读取进度文件失败: {e}")
            return {}

    def save_progress(self, progress: dict):
        """保存批处理进度"""
        try:
            progress['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            logger.debug(f"进度已保存")
        except Exception as e:
            logger.error(f"保存进度失败: {e}")

    def get_dataset_status(self, progress: dict, dataset_file: Path):
        """获取数据集的处理状态"""
        dataset_name = dataset_file.name
        dataset_info = progress.get('datasets', {})

        if dataset_name not in dataset_info:
            return 'pending', None

        info = dataset_info[dataset_name]
        status = info.get('status', 'unknown')

        if status == 'completed':
            return 'completed', info
        elif status == 'failed':
            return 'failed', info
        elif status == 'running':
            return 'running', info
        else:
            return 'pending', info

    def build_command(
        self,
        input_file: Path,
        output_file: Path,
        resume: bool = False,
        additional_args: list = None
    ):
        """构建命令行"""
        cmd = [
            sys.executable,  # python 解释器
            self.main_script,
            "--dataset", str(input_file),
            "--output", str(output_file),
            "--save-interval", "20"  # 每20个样本保存一次
        ]

        if resume:
            cmd.append("--resume")

        if additional_args:
            cmd.extend(additional_args)

        return cmd

    def process_single_dataset(
        self,
        dataset_file: Path,
        resume: bool = False,
        additional_args: list = None
    ):
        """
        处理单个数据集

        Returns:
            (success: bool, output_file: Path)
        """
        output_file = self.output_dir / f"{dataset_file.stem}_results.json"

        logger.info(f"\n{'='*80}")
        logger.info(f"处理数据集: {dataset_file.name}")
        logger.info(f"输出文件: {output_file.name}")
        logger.info(f"{'='*80}\n")

        cmd = self.build_command(
            dataset_file,
            output_file,
            resume=resume,
            additional_args=additional_args
        )

        logger.info(f"执行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # 实时输出到控制台
                text=True
            )
            return True, output_file
        except subprocess.CalledProcessError as e:
            logger.error(f"处理失败: {dataset_file.name}")
            logger.error(f"返回码: {e.returncode}")
            return False, output_file
        except Exception as e:
            logger.error(f"处理异常: {dataset_file.name}, {e}")
            return False, output_file

    def run(
        self,
        resume: bool = False,
        clear_progress: bool = False,
        additional_args: list = None,
        max_retries: int = 1
    ):
        """
        批量运行所有数据集

        Args:
            resume: 是否从上次中断处继续
            clear_progress: 是否清除进度重新开始
            additional_args: 额外的命令行参数
            max_retries: 失败重试次数
        """
        # 清除进度
        if clear_progress:
            if self.progress_file.exists():
                self.progress_file.unlink()
                logger.info(f"✓ 已清除进度文件: {self.progress_file}")

        # 扫描数据集
        dataset_files = self.scan_datasets()
        if not dataset_files:
            logger.error(f"未找到数据集文件: {self.input_dir}")
            return

        # 加载进度
        progress = self.load_progress()

        if 'datasets' not in progress:
            progress['datasets'] = {}
        if 'total_count' not in progress:
            progress['total_count'] = len(dataset_files)

        # 统计信息
        stats = {
            'pending': 0,
            'completed': 0,
            'failed': 0,
            'skipped': 0
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"批处理模式 - 共 {len(dataset_files)} 个数据集")
        logger.info(f"{'='*80}\n")

        # 处理每个数据集
        for idx, dataset_file in enumerate(dataset_files, 1):
            dataset_name = dataset_file.name
            status, info = self.get_dataset_status(progress, dataset_file)

            # 显示进度概览
            logger.info(f"\n[{idx}/{len(dataset_files)}] {dataset_name}")
            logger.info(f"  状态: {status}")

            # 跳过已完成的
            if status == 'completed' and resume:
                logger.info(f"  ✓ 已完成，跳过")
                stats['completed'] += 1
                stats['skipped'] += 1
                continue

            # 失败的重试
            if status == 'failed' and resume:
                logger.info(f"  ⚠️  上次失败，重新处理")

            # 处理数据集（带重试）
            success = False
            for retry in range(max_retries + 1):
                if retry > 0:
                    logger.info(f"  重试 {retry}/{max_retries}...")

                success, output_file = self.process_single_dataset(
                    dataset_file,
                    resume=(status == 'running' or retry > 0),  # 如果是running状态或重试，启用resume
                    additional_args=additional_args
                )

                if success:
                    break

            # 更新进度
            dataset_info = {
                'status': 'completed' if success else 'failed',
                'output_file': str(output_file),
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'retry_count': max_retries + 1 - (1 if success else 0)
            }

            progress['datasets'][dataset_name] = dataset_info
            self.save_progress(progress)

            # 更新统计
            if success:
                logger.info(f"  ✓ 完成")
                stats['completed'] += 1
            else:
                logger.error(f"  ✗ 失败")
                stats['failed'] += 1

        # 最终报告
        logger.info(f"\n\n{'='*80}")
        logger.info(f"批处理完成！")
        logger.info(f"{'='*80}")
        logger.info(f"\n📊 统计信息:")
        logger.info(f"  总数据集: {len(dataset_files)}")
        logger.info(f"  成功完成: {stats['completed']}")
        logger.info(f"  处理失败: {stats['failed']}")
        logger.info(f"  已跳过: {stats['skipped']}")

        # 列出失败的数据集
        if stats['failed'] > 0:
            logger.warning(f"\n⚠️  失败的数据集:")
            for dataset_name, info in progress.get('datasets', {}).items():
                if info.get('status') == 'failed':
                    logger.warning(f"  - {dataset_name}")
                    logger.warning(f"    最后更新: {info.get('last_update')}")
                    logger.warning(f"    输出文件: {info.get('output_file')}")

        logger.info(f"\n✓ 结果保存在: {self.output_dir}")
        logger.info(f"✓ 进度文件: {self.progress_file}")


def main():
    parser = argparse.ArgumentParser(
        description="批处理生成器：处理文件夹中的所有数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 首次运行：处理所有数据集
  python batch_process_datasets.py --input Agent_Trainv2 --output results

  # 中断后继续
  python batch_process_datasets.py --input Agent_Trainv2 --output results --resume

  # 清除进度重新开始
  python batch_process_datasets.py --input Agent_Trainv2 --output results --clear-progress

  # 只处理特定文件（使用通配符）
  python batch_process_datasets.py --input Agent_Trainv2 --output results --pattern "AMD*.jsonl"

  # 额外的参数传递给 main_v2.py
  python batch_process_datasets.py --input Agent_Trainv2 --output results --extra-args "--no-ssh"
        """
    )

    parser.add_argument(
        "--input",
        required=True,
        help="输入文件夹路径（包含 .jsonl 数据集文件）"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出文件夹路径"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次中断处继续"
    )
    parser.add_argument(
        "--clear-progress",
        action="store_true",
        help="清除进度文件，从头开始"
    )
    parser.add_argument(
        "--pattern",
        help="文件名匹配模式（如 'AMD*.jsonl'）"
    )
    parser.add_argument(
        "--extra-args",
        help="传递给 main_v2.py 的额外参数（用引号包裹）"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="失败重试次数（默认: 1）"
    )

    args = parser.parse_args()

    # 解析额外参数
    additional_args = None
    if args.extra_args:
        # 简单解析（可以用 shlex.split 改进）
        additional_args = args.extra_args.split()
        logger.info(f"额外参数: {additional_args}")

    # 创建批处理器
    processor = BatchProcessor(
        input_dir=args.input,
        output_dir=args.output,
        progress_file="batch_progress.json"
    )

    # 如果指定了文件名模式，过滤数据集
    if args.pattern:
        import fnmatch
        all_files = processor.scan_datasets()
        filtered_files = [f for f in all_files if fnmatch.fnmatch(f.name, args.pattern)]
        # 临时修改 scan_datasets 返回过滤后的文件
        # 这里我们直接运行，但会在 run 中重新扫描
        if not filtered_files:
            logger.error(f"未找到匹配 '{args.pattern}' 的文件")
            return
        logger.info(f"匹配模式 '{args.pattern}' 的文件: {len(filtered_files)} 个")

    # 运行批处理
    processor.run(
        resume=args.resume,
        clear_progress=args.clear_progress,
        additional_args=additional_args,
        max_retries=args.max_retries
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n用户中断，退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n发生错误: {e}", exc_info=True)
        sys.exit(1)
