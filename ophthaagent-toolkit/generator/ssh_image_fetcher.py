"""
SSH远程图片获取器
从SSH服务器下载数据集图片到本地缓存
支持密钥认证和密码认证
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict
import hashlib
import subprocess
import time

logger = logging.getLogger(__name__)

# 检查是否安装了paramiko
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    logger.warning("paramiko未安装，密码认证功能不可用。请运行: pip install paramiko")


class SSHImageFetcher:
    """通过SSH从远程服务器获取图片"""

    def __init__(
        self,
        ssh_host: str = None,
        ssh_user: str = None,
        ssh_password: str = None,
        ssh_port: int = 22,
        remote_base_path: str = None,
        local_cache_dir: str = "./image_cache",
        ssh_key_path: str = None,
        use_scp: bool = True
    ):
        """
        初始化SSH图片获取器

        Args:
            ssh_host: SSH服务器地址 (如 "192.168.1.100")
            ssh_user: SSH用户名
            ssh_password: SSH密码 (可选，用于密码认证)
            ssh_port: SSH端口 (默认22)
            remote_base_path: 远程数据集根目录 (如 "<ANON_ABS_PATH>")
            local_cache_dir: 本地缓存目录
            ssh_key_path: SSH密钥路径 (可选，优先使用密钥认证)
            use_scp: 是否使用scp (否则使用rsync)
        """
        self.ssh_host = ssh_host or os.getenv("SSH_HOST")
        self.ssh_user = ssh_user or os.getenv("SSH_USER")
        self.ssh_password = ssh_password or os.getenv("SSH_PASSWORD")
        self.ssh_port = ssh_port or int(os.getenv("SSH_PORT", "22"))
        self.remote_base_path = remote_base_path or os.getenv("REMOTE_BASE_PATH", "<ANON_ABS_PATH>")
        self.local_cache_dir = Path(local_cache_dir)
        self.ssh_key_path = ssh_key_path or os.getenv("SSH_KEY_PATH")
        self.use_scp = use_scp

        # 创建缓存目录
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # 验证SSH配置
        if not self.ssh_host or not self.ssh_user:
            logger.warning("SSH配置不完整，将使用本地路径模式")
            self.enabled = False
        else:
            self.enabled = True
            # 确定认证方式
            if self.ssh_key_path:
                self.auth_method = "key"
                logger.info(f"SSH图片获取器初始化: {self.ssh_user}@{self.ssh_host} (密钥认证)")
            elif self.ssh_password:
                self.auth_method = "password"
                logger.info(f"SSH图片获取器初始化: {self.ssh_user}@{self.ssh_host} (密码认证)")
            else:
                self.auth_method = "agent"
                logger.info(f"SSH图片获取器初始化: {self.ssh_user}@{self.ssh_host} (ssh-agent)")

        # SSH连接池 (用于密码认证)
        self.ssh_client = None
        self.sftp_client = None

        # 下载统计
        self.download_count = 0
        self.cache_hit_count = 0
        self.total_bytes = 0

    def _get_ssh_client(self):
        """获取SSH客户端连接 (用于密码认证)"""
        if not PARAMIKO_AVAILABLE:
            raise RuntimeError("paramiko未安装，无法使用密码认证。请运行: pip install paramiko")

        if self.ssh_client is None:
            try:
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                # 连接参数
                connect_kwargs = {
                    'hostname': self.ssh_host,
                    'port': self.ssh_port,
                    'username': self.ssh_user,
                    'timeout': 10
                }

                # 根据认证方式连接
                if self.auth_method == "key" and self.ssh_key_path:
                    connect_kwargs['key_filename'] = self.ssh_key_path
                elif self.auth_method == "password" and self.ssh_password:
                    connect_kwargs['password'] = self.ssh_password

                self.ssh_client.connect(**connect_kwargs)
                logger.debug("SSH客户端连接成功")

            except Exception as e:
                logger.error(f"SSH连接失败: {e}")
                self.ssh_client = None
                raise

        return self.ssh_client

    def _get_sftp_client(self):
        """获取SFTP客户端"""
        if self.sftp_client is None:
            ssh_client = self._get_ssh_client()
            self.sftp_client = ssh_client.open_sftp()
        return self.sftp_client

    def close(self):
        """关闭SSH连接"""
        if self.sftp_client:
            self.sftp_client.close()
            self.sftp_client = None
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
        logger.debug("SSH连接已关闭")

    def get_local_path(self, relative_path: str) -> str:
        """
        获取图片的本地路径，如果不存在则从SSH下载

        Args:
            relative_path: 相对路径 (如 "15_APTOS/train_images/c4a8f2fcf6e8.png")

        Returns:
            本地缓存路径
        """
        # 清理路径
        relative_path = relative_path.strip()

        # 计算本地缓存路径
        local_path = self.local_cache_dir / relative_path

        # 如果本地已存在，直接返回
        if local_path.exists():
            self.cache_hit_count += 1
            logger.debug(f"✓ 缓存命中: {relative_path}")
            return str(local_path)

        # 如果SSH未启用，尝试使用本地路径
        if not self.enabled:
            # 尝试多个可能的本地路径
            possible_paths = [
                Path(relative_path),  # 相对于当前目录
                Path("..") / relative_path,  # 上级目录
                Path("../..") / relative_path,  # 再上级
                Path(self.remote_base_path) / relative_path,  # 配置的基础路径
            ]

            for path in possible_paths:
                if path.exists():
                    logger.info(f"✓ 使用本地路径: {path}")
                    return str(path)

            logger.error(f"✗ 图片不存在: {relative_path} (SSH未启用)")
            raise FileNotFoundError(f"图片不存在且SSH未配置: {relative_path}")

        # 从SSH服务器下载
        try:
            self._download_from_ssh(relative_path, local_path)
            self.download_count += 1
            logger.info(f"✓ SSH下载完成: {relative_path}")
            return str(local_path)

        except Exception as e:
            logger.error(f"✗ SSH下载失败: {relative_path}, {e}")
            raise

    def _download_from_ssh(self, relative_path: str, local_path: Path):
        """从SSH服务器下载单个文件"""
        # 创建本地目录
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # 构建远程路径
        remote_path = f"{self.remote_base_path}/{relative_path}"

        # 如果使用密码认证，优先使用SFTP
        if self.auth_method == "password":
            self._download_with_sftp(remote_path, local_path)
        elif self.use_scp:
            # 使用scp下载 (密钥认证或ssh-agent)
            self._download_with_scp(remote_path, local_path)
        else:
            # 使用rsync下载
            self._download_with_rsync(remote_path, local_path)

    def _download_with_sftp(self, remote_path: str, local_path: Path):
        """使用SFTP下载文件 (用于密码认证)"""
        try:
            sftp = self._get_sftp_client()

            logger.debug(f"执行SFTP下载: {remote_path} -> {local_path}")
            start_time = time.time()

            # 下载文件
            sftp.get(remote_path, str(local_path))

            # 统计
            elapsed = time.time() - start_time
            file_size = local_path.stat().st_size
            self.total_bytes += file_size

            logger.debug(f"SFTP下载完成: {file_size / 1024:.1f}KB, 耗时: {elapsed:.2f}s")

        except Exception as e:
            raise RuntimeError(f"SFTP下载失败: {e}")

    def _download_with_scp(self, remote_path: str, local_path: Path):
        """使用scp下载文件 (用于密钥认证)"""
        # 构建scp命令
        remote_url = f"{self.ssh_user}@{self.ssh_host}:{remote_path}"

        cmd = ["scp", "-P", str(self.ssh_port)]

        # 添加SSH密钥
        if self.ssh_key_path:
            cmd.extend(["-i", self.ssh_key_path])

        # 添加源和目标
        cmd.extend([remote_url, str(local_path)])

        logger.debug(f"执行SCP: {' '.join(cmd)}")

        # 执行下载
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60秒超时
        )

        if result.returncode != 0:
            raise RuntimeError(f"SCP下载失败: {result.stderr}")

        # 统计
        elapsed = time.time() - start_time
        file_size = local_path.stat().st_size
        self.total_bytes += file_size

        logger.debug(f"SCP下载完成: {file_size / 1024:.1f}KB, 耗时: {elapsed:.2f}s")

    def _download_with_rsync(self, remote_path: str, local_path: Path):
        """使用rsync下载文件"""
        # 构建rsync命令
        remote_url = f"{self.ssh_user}@{self.ssh_host}:{remote_path}"

        cmd = ["rsync", "-avz", "-e", f"ssh -p {self.ssh_port}"]

        # 添加SSH密钥
        if self.ssh_key_path:
            cmd[-1] = f"ssh -p {self.ssh_port} -i {self.ssh_key_path}"

        # 添加源和目标
        cmd.extend([remote_url, str(local_path)])

        logger.debug(f"执行rsync: {' '.join(cmd)}")

        # 执行下载
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(f"rsync下载失败: {result.stderr}")

        elapsed = time.time() - start_time
        file_size = local_path.stat().st_size
        self.total_bytes += file_size

        logger.debug(f"下载完成: {file_size / 1024:.1f}KB, 耗时: {elapsed:.2f}s")

    def batch_download(self, relative_paths: list[str], max_workers: int = 3) -> Dict[str, str]:
        """
        批量下载图片

        Args:
            relative_paths: 相对路径列表
            max_workers: 最大并发数

        Returns:
            {relative_path: local_path} 映射字典
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        result_map = {}
        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_path = {
                executor.submit(self.get_local_path, path): path
                for path in relative_paths
            }

            # 收集结果
            for future in as_completed(future_to_path):
                relative_path = future_to_path[future]
                try:
                    local_path = future.result()
                    result_map[relative_path] = local_path
                except Exception as e:
                    logger.error(f"批量下载失败: {relative_path}, {e}")
                    failed.append(relative_path)

        if failed:
            logger.warning(f"批量下载有 {len(failed)} 个文件失败")

        return result_map

    def clear_cache(self):
        """清空本地缓存"""
        import shutil
        if self.local_cache_dir.exists():
            shutil.rmtree(self.local_cache_dir)
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("缓存已清空")

    def get_stats(self) -> Dict:
        """获取下载统计"""
        cache_size = sum(
            f.stat().st_size
            for f in self.local_cache_dir.rglob("*")
            if f.is_file()
        )

        return {
            "enabled": self.enabled,
            "ssh_host": self.ssh_host,
            "downloads": self.download_count,
            "cache_hits": self.cache_hit_count,
            "total_bytes": self.total_bytes,
            "cache_size_mb": cache_size / (1024 * 1024),
            "cache_dir": str(self.local_cache_dir)
        }

    def test_connection(self) -> bool:
        """测试SSH连接"""
        if not self.enabled:
            logger.warning("SSH未配置")
            return False

        # 如果使用密码认证，使用paramiko测试
        if self.auth_method == "password":
            try:
                ssh_client = self._get_ssh_client()
                # 执行简单命令测试
                stdin, stdout, stderr = ssh_client.exec_command("echo 'SSH connection OK'")
                output = stdout.read().decode().strip()

                if output == "SSH connection OK":
                    logger.info("✓ SSH连接测试成功 (密码认证)")
                    return True
                else:
                    logger.error(f"✗ SSH连接测试失败: {stderr.read().decode()}")
                    return False

            except Exception as e:
                logger.error(f"✗ SSH连接测试异常: {e}")
                return False

        # 使用密钥认证或ssh-agent，用ssh命令测试
        cmd = ["ssh", "-p", str(self.ssh_port)]

        if self.ssh_key_path:
            cmd.extend(["-i", self.ssh_key_path])

        cmd.extend([
            f"{self.ssh_user}@{self.ssh_host}",
            "echo 'SSH connection OK'"
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info("✓ SSH连接测试成功")
                return True
            else:
                logger.error(f"✗ SSH连接测试失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"✗ SSH连接测试异常: {e}")
            return False


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 从环境变量或参数创建fetcher
    fetcher = SSHImageFetcher(
        ssh_host="example.com",  # 替换为实际服务器
        ssh_user="username",
        remote_base_path="<ANON_ABS_PATH>",
        local_cache_dir="./image_cache"
    )

    # 测试连接
    print("\n=== 测试SSH连接 ===")
    fetcher.test_connection()

    # 测试下载单个文件
    print("\n=== 测试单个下载 ===")
    try:
        local_path = fetcher.get_local_path("15_APTOS/train_images/c4a8f2fcf6e8.png")
        print(f"本地路径: {local_path}")
    except Exception as e:
        print(f"下载失败: {e}")

    # 显示统计
    print("\n=== 下载统计 ===")
    stats = fetcher.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
