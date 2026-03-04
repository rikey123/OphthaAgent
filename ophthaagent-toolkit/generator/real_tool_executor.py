"""
真实工具执行器
调用项目中的实际工具，支持缓存和并行执行
"""
import sys
import os
import json
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class RealToolExecutor:
    """真实工具执行器，调用项目中的实际工具"""

    # 工具名称映射表（生成器使用的名称 -> 实际函数名称）
    TOOL_MAPPING = {
        # DR相关
        "DR_Grading": "DRgrading_by_DINO",  # ✅ DINO模型（新版）
        "detect_lesions": "segment_by_unet",  # ✅ UNet病灶分割（新版）
        "fundus_lesion_segmentation": "fundus_lesion_segmentation",  # 保留向后兼容
        "segment_by_unet": "segment_by_unet",  # UNet新版
        "DR_segmentation_anchor_ETDRS": "DR_segmentation_anchor_ETDRS",

        # AMD相关
        "AMD_predict": "AMD_predict_fundus_by_deepseenet",
        "AMD_predict_fundus": "AMD_predict_fundus_by_deepseenet",
        "AMD_predict_fundus_by_deepseenet": "AMD_predict_fundus_by_deepseenet",

        # 青光眼相关（使用AutoMorphalyzer）
        "segment_by_AutoMorphalyzer": "segment_by_AutoMorphalyzer",
        "glaucoma_segment": "segment_by_AutoMorphalyzer",  # 别名

        # OCT相关
        "Lesion_predict_OCT": "Lesion_predict_OCT_by_opticnet",
        "Lesion_predict_OCT_by_opticnet": "Lesion_predict_OCT_by_opticnet",

        # DME相关
        "dme_risk_assess": "dme_risk_assessment",
        "dme_risk_assessment": "dme_risk_assessment",

        # 图像质量与预处理
        "quality_assess": "quality_assess_by_fit",
        "quality_assess_by_fit": "quality_assess_by_fit",
        "enhance_image": "enhance_fundus_image",
        "enhance_fundus_image": "enhance_fundus_image",
        "enhance_by_fit": "enhance_by_fit",
        "crop_by_fit": "crop_by_fit",

        # 解剖结构定位
        "fovea_od_localize": "fov_od_localization_by_fit",
        "fov_od_localization_by_fit": "fov_od_localization_by_fit",
        "vessel_segment": "vessel_segment_by_fit",
        "vessel_segment_by_fit": "vessel_segment_by_fit",

        # 新增Crop工具
        "crop_roi": "crop_fundus_roi",
        "crop_fundus_roi": "crop_fundus_roi",

        # RAG查询
        "rag_query": "rag_query"
    }

    def __init__(
        self,
        tools_interface_path: str = None,
        enable_cache: bool = True,
        cache_dir: str = "./tool_cache"
    ):
        """
        初始化工具执行器

        Args:
            tools_interface_path: tools_interface.py所在目录路径
            enable_cache: 是否启用缓存
            cache_dir: 缓存目录
        """
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir)
        self.tool_cache = {}  # 内存缓存

        if enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()

        # 动态导入tools_interface模块
        if tools_interface_path is None:
            # 默认路径
            tools_interface_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "agents",
                "fundus_tools_agent"
            )

        sys.path.insert(0, os.path.abspath(tools_interface_path))

        try:
            # 导入多个tools_interface模块
            self.tools_dict = {}

            # 1. 主要的 tools_interface.py
            if os.path.exists(os.path.join(tools_interface_path, "tools_interface.py")):
                import tools_interface as ti
                self.tools = ti
                # 添加所有函数到字典
                for attr_name in dir(ti):
                    if not attr_name.startswith('_'):
                        self.tools_dict[attr_name] = getattr(ti, attr_name)

                # ✅ 新版DR分级工具（DINO）
                if hasattr(ti, 'DRgrading_by_DINO'):
                    logger.info("  ✓ 加载DR分级工具: DRgrading_by_DINO")

                logger.info(f"✓ 加载 tools_interface.py")
            else:
                self.tools = None
                logger.warning(f"未找到 tools_interface.py")

            # 2. tools_interface1.py (图像增强和病灶分割)
            tools1_path = os.path.join(tools_interface_path, "tools", "tools_interface1.py")
            if os.path.exists(tools1_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("tools_interface1", tools1_path)
                ti1 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ti1)

                # 只导入图像增强和病灶分割工具（老版AutoMorph已废弃）
                if hasattr(ti1, 'enhance_fundus_image'):
                    self.tools_dict['enhance_fundus_image'] = ti1.enhance_fundus_image
                if hasattr(ti1, 'fundus_lesion_segmentation'):
                    self.tools_dict['fundus_lesion_segmentation'] = ti1.fundus_lesion_segmentation
                logger.info(f"✓ 加载 tools_interface1.py (图像增强/病灶分割)")
            else:
                logger.warning(f"未找到 tools_interface1.py")

            # 3. tool_interface3.py (DR, UNet, AutoMorphalyzer)
            tools3_path = os.path.join(tools_interface_path, "tool_interface3.py")
            if os.path.exists(tools3_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("tool_interface3", tools3_path)
                ti3 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ti3)

                if hasattr(ti3, 'DR_Grading'):
                    self.tools_dict['DR_Grading'] = ti3.DR_Grading
                if hasattr(ti3, 'segment_by_unet'):
                    self.tools_dict['segment_by_unet'] = ti3.segment_by_unet
                    logger.info("  ✓ 加载病灶分割工具: segment_by_unet (UNet)")
                # 新增病灶分布可视化工具
                if hasattr(ti3, 'DR_segmentation_anchor_ETDRS'):
                    self.tools_dict['DR_segmentation_anchor_ETDRS'] = ti3.DR_segmentation_anchor_ETDRS
                    logger.info("  ✓ 加载病灶分布可视化工具: DR_segmentation_anchor_ETDRS")
                logger.info(f"✓ 加载 tool_interface3.py (DR/UNet)")
            else:
                logger.warning(f"未找到 tool_interface3.py")

            # 4. tool_interface4.py (AMD, OCT)
            tools4_path = os.path.join(tools_interface_path, "tool_interface4.py")
            if os.path.exists(tools4_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("tool_interface4", tools4_path)
                ti4 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ti4)

                if hasattr(ti4, 'AMD_predict_fundus_by_deepseenet'):
                    self.tools_dict['AMD_predict_fundus_by_deepseenet'] = ti4.AMD_predict_fundus_by_deepseenet
                if hasattr(ti4, 'Lesion_predict_OCT_by_opticnet'):
                    self.tools_dict['Lesion_predict_OCT_by_opticnet'] = ti4.Lesion_predict_OCT_by_opticnet
                if hasattr(ti4, 'segment_by_AutoMorphalyzer'):
                    self.tools_dict['segment_by_AutoMorphalyzer'] = ti4.segment_by_AutoMorphalyzer
                if hasattr(ti4, 'dme_risk_assessment'):
                    self.tools_dict['dme_risk_assessment'] = ti4.dme_risk_assessment
                    logger.info(f"✓ 加载 dme_risk_assessment")
                logger.info(f"✓ 加载 tool_interface4.py (AMD/OCT)")
            else:
                logger.warning(f"未找到 tool_interface4.py")

            # 5. 导入crop_roi模块
            crop_roi_path = os.path.join(tools_interface_path, "preprocessing", "crop_roi.py")
            if os.path.exists(crop_roi_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("crop_roi", crop_roi_path)
                crop_roi_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(crop_roi_module)
                self.tools_dict['crop_fundus_roi'] = crop_roi_module.crop_fundus_roi
                logger.info("✓ 加载 crop_roi.py")
            else:
                logger.warning("未找到 crop_roi.py")

            logger.info(f"总计加载 {len(self.tools_dict)} 个工具函数")

        except Exception as e:
            logger.error(f"加载工具接口失败: {e}", exc_info=True)
            self.tools = None
            self.tools_dict = {}

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict:
        """
        执行真实工具并返回结果

        Args:
            tool_name: 工具名称（生成器使用的标准名称）
            args: 工具参数

        Returns:
            dict: 工具执行结果
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(tool_name, args)
        logger.info(f"🔍 [DEBUG] 生成缓存键: {cache_key}")

        # 检查缓存
        if self.enable_cache and cache_key in self.tool_cache:
            logger.info(f"✓ 使用缓存结果: {tool_name}")
            cached_result = self.tool_cache[cache_key]
            logger.info(f"🔍 [DEBUG] 缓存命中，缓存结果的键: {list(cached_result.keys()) if isinstance(cached_result, dict) else 'N/A'}")
            
            # ✅ 即使是缓存结果，也要检查并添加标注图标记
            if tool_name == "DR_segmentation_anchor_ETDRS":
                if isinstance(cached_result, dict) and cached_result.get("status") == "success":
                    # 检查 output_visualization_path（可能在顶层、result 或 results 中）
                    output_visualization_path = None
                    
                    # 先检查顶层
                    if "output_visualization_path" in cached_result:
                        output_visualization_path = cached_result["output_visualization_path"]
                        logger.info(f"  ✓ 在顶层找到 output_visualization_path")
                    else:
                        # 检查 'result'（单数）- 旧格式
                        if "result" in cached_result:
                            result_data = cached_result.get("result", {})
                            if isinstance(result_data, dict) and "output_visualization_path" in result_data:
                                output_visualization_path = result_data["output_visualization_path"]
                                logger.info(f"  ✓ 在 result (单数) 中找到 output_visualization_path")
                        
                        # 检查 'results'（复数）- 新格式
                        if not output_visualization_path and "results" in cached_result:
                            results_data = cached_result.get("results", {})
                            if isinstance(results_data, dict) and "output_visualization_path" in results_data:
                                output_visualization_path = results_data["output_visualization_path"]
                                logger.info(f"  ✓ 在 results (复数) 中找到 output_visualization_path")
                    
                    if output_visualization_path:
                        cached_result["_is_annotated_image"] = True
                        cached_result["_annotated_image_path"] = output_visualization_path
                        logger.info(f"✓ 为缓存结果添加标注图标记: {output_visualization_path}")
            
            return cached_result

        # 映射工具名称
        if tool_name not in self.TOOL_MAPPING:
            logger.warning(f"未知工具: {tool_name}，尝试直接调用")
            real_tool_name = tool_name
        else:
            real_tool_name = self.TOOL_MAPPING[tool_name]

        # 执行工具
        logger.info(f"⚙ 执行真实工具: {tool_name} -> {real_tool_name}")
        logger.debug(f"  参数: {args}")

        try:
            # 特殊处理rag_query（可能需要不同的调用方式）
            if tool_name == "rag_query":
                result = self._execute_rag_query(args)
            # 从tools_dict查找并调用工具
            elif real_tool_name in self.tools_dict:
                tool_func = self.tools_dict[real_tool_name]
                result = tool_func(**args)
                logger.info(f"🔍 [DEBUG] 工具函数返回的 result 类型: {type(result)}")
                if isinstance(result, dict):
                    logger.info(f"🔍 [DEBUG] 工具函数返回的 result 键: {list(result.keys())}")
                    logger.info(f"🔍 [DEBUG] 工具函数返回的 result 内容: {str(result)}")
            # 尝试从tools模块获取（向后兼容）
            elif self.tools and hasattr(self.tools, real_tool_name):
                tool_func = getattr(self.tools, real_tool_name)
                result = tool_func(**args)
                logger.info(f"🔍 [DEBUG] 工具函数返回的 result 类型: {type(result)}")
                if isinstance(result, dict):
                    logger.info(f"🔍 [DEBUG] 工具函数返回的 result 键: {list(result.keys())}")
                    logger.info(f"🔍 [DEBUG] 工具函数返回的 result 内容: {str(result)}")
            else:
                # 模拟模式：返回占位符结果
                logger.warning(f"工具 {real_tool_name} 不可用，使用模拟结果")
                result = self._generate_mock_result(tool_name, args)

            # ✅ 特殊处理：为 DR_segmentation_anchor_ETDRS 添加标注图标记
            if tool_name == "DR_segmentation_anchor_ETDRS":
                logger.info(f"🔍 [DEBUG] 开始检查 DR_segmentation_anchor_ETDRS 结果...")
                logger.info(f"  [DEBUG] result 类型: {type(result)}")
                logger.info(f"  [DEBUG] result 是否为 dict: {isinstance(result, dict)}")
                
                if isinstance(result, dict):
                    logger.info(f"  [DEBUG] result 的所有键: {list(result.keys())}")
                    logger.info(f"  [DEBUG] result.get('status'): {result.get('status')}")
                    logger.info(f"  [DEBUG] 'status' == 'success': {result.get('status') == 'success'}")
                
                if isinstance(result, dict) and result.get("status") == "success":
                    logger.info(f"  [DEBUG] 进入 success 分支")
                    # 检查 output_visualization_path（可能在顶层、result 或 results 中）
                    output_visualization_path = None
                    
                    # 先检查顶层
                    logger.info(f"  [DEBUG] 检查顶层 'output_visualization_path'...")
                    if "output_visualization_path" in result:
                        output_visualization_path = result["output_visualization_path"]
                        logger.info(f"  ✓ 在顶层找到 output_visualization_path")
                    else:
                        logger.info(f"  [DEBUG] 顶层未找到 'output_visualization_path'")
                        # 检查 'result'（单数）- 旧格式
                        logger.info(f"  [DEBUG] 检查 'result' (单数)...")
                        if "result" in result:
                            result_data = result.get("result", {})
                            logger.info(f"  [DEBUG] result_data 类型: {type(result_data)}")
                            logger.info(f"  [DEBUG] result_data 键: {list(result_data.keys()) if isinstance(result_data, dict) else 'N/A'}")
                            if isinstance(result_data, dict):
                                # 先检查 result_data 中的 output_visualization_path
                                if "output_visualization_path" in result_data:
                                    output_visualization_path = result_data["output_visualization_path"]
                                    logger.info(f"  ✓ 在 result (单数) 中找到 output_visualization_path")
                                # 再检查 result_data 中的 results 键
                                elif "results" in result_data:
                                    results_data = result_data.get("results", {})
                                    logger.info(f"  [DEBUG] results_data 类型: {type(results_data)}")
                                    logger.info(f"  [DEBUG] results_data 键: {list(results_data.keys()) if isinstance(results_data, dict) else 'N/A'}")
                                    if isinstance(results_data, dict) and "output_visualization_path" in results_data:
                                        output_visualization_path = results_data["output_visualization_path"]
                                        logger.info(f"  ✓ 在 result.results (复数) 中找到 output_visualization_path")
                        
                        # 检查顶层 'results'（复数）- 新格式
                        logger.info(f"  [DEBUG] 检查顶层 'results' (复数)...")
                        if not output_visualization_path and "results" in result:
                            results_data = result.get("results", {})
                            logger.info(f"  [DEBUG] results_data 类型: {type(results_data)}")
                            logger.info(f"  [DEBUG] results_data 键: {list(results_data.keys()) if isinstance(results_data, dict) else 'N/A'}")
                            if isinstance(results_data, dict) and "output_visualization_path" in results_data:
                                output_visualization_path = results_data["output_visualization_path"]
                                logger.info(f"  ✓ 在顶层 results (复数) 中找到 output_visualization_path")
                    
                    logger.info(f"  [DEBUG] 最终 output_visualization_path: {output_visualization_path}")
                    if output_visualization_path:
                        result["_is_annotated_image"] = True
                        result["_annotated_image_path"] = output_visualization_path
                        logger.info(f"✓ 标记 DR_segmentation_anchor_ETDRS 输出为标注图: {output_visualization_path}")
                    else:
                        logger.warning(f"⚠️ DR_segmentation_anchor_ETDRS 结果中未找到 output_visualization_path")

            # 缓存结果
            if self.enable_cache:
                self.tool_cache[cache_key] = result
                self._save_to_persistent_cache(cache_key, result)

            logger.info(f"✓ 工具 {tool_name} 执行成功")
            return result

        except Exception as e:
            logger.error(f"✗ 工具 {tool_name} 执行失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tool_name": tool_name,
                "args": args
            }

    def batch_execute(
        self,
        tool_calls: List[Dict[str, Any]],
        max_workers: int = 3
    ) -> List[Dict]:
        """
        批量并行执行工具

        Args:
            tool_calls: 工具调用列表，每个元素格式: {"tool_name": str, "args": dict}
            max_workers: 最大并发数

        Returns:
            list: 执行结果列表
        """
        logger.info(f"批量执行 {len(tool_calls)} 个工具...")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_call = {
                executor.submit(
                    self.execute_tool,
                    call["tool_name"],
                    call["args"]
                ): call
                for call in tool_calls
            }

            # 收集结果
            for future in as_completed(future_to_call):
                call = future_to_call[future]
                try:
                    result = future.result()
                    results.append({
                        "tool_name": call["tool_name"],
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"工具执行异常: {call['tool_name']}, {e}")
                    results.append({
                        "tool_name": call["tool_name"],
                        "result": {
                            "status": "error",
                            "error": str(e)
                        }
                    })

        return results

    def _generate_cache_key(self, tool_name: str, args: Dict) -> str:
        """生成缓存键"""
        # 将参数转换为稳定的字符串
        args_str = json.dumps(args, sort_keys=True)
        cache_str = f"{tool_name}:{args_str}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _load_persistent_cache(self):
        """加载持久化缓存"""
        cache_file = self.cache_dir / "tool_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.tool_cache = pickle.load(f)
                logger.info(f"加载缓存: {len(self.tool_cache)} 条记录")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                self.tool_cache = {}

    def _save_to_persistent_cache(self, key: str, value: Dict):
        """保存到持久化缓存"""
        cache_file = self.cache_dir / "tool_cache.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.tool_cache, f)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def _execute_rag_query(self, args: Dict) -> Dict:
        """执行RAG查询（需要根据实际RAG系统调整）"""
        # 这里需要根据你的RAG系统实现
        # 暂时返回模拟结果
        query = args.get("query", "")
        return {
            "status": "success",
            "retrieved_context": f"[模拟RAG响应] 关于'{query}'的医学知识..."
        }

    def _generate_mock_result(self, tool_name: str, args: Dict) -> Dict:
        """生成模拟结果（用于测试）"""
        mock_results = {
            "DR_Grading": {
                "status": "success",
                "result": {
                    "prediction": {
                        "class_id": 2,
                        "class_name": "中度",
                        "confidence": 0.85
                    },
                    "probabilities": {
                        "0_normal": 0.05,
                        "1_mild": 0.10,
                        "2_moderate": 0.75,
                        "3_severe": 0.08,
                        "4_proliferative": 0.02
                    },
                    "model_info": {
                        "model_name": "vit_base_patch14_dinov2.lvd142m",
                        "input_size": 518
                    }
                }
            },
            "detect_lesions": {
                "status": "success",
                "result": {
                    "分析时间": "2026-01-17 21:07:37",
                    "输入图像": "/path/to/test.jpg",
                    "结果目录": "/path/to/results",
                    "分析结果": {
                        "微动脉瘤分割": {
                            "状态": "微动脉瘤分割完成",
                            "指标": {
                                "微动脉瘤数量": 8,
                                "微动脉瘤区域占比": 0.005
                            }
                        },
                        "出血分割": {
                            "状态": "出血分割完成",
                            "指标": {
                                "出血区域占比": 0.023,
                                "出血区域总大小": 4045.0
                            }
                        },
                        "硬渗出物分割": {
                            "状态": "硬渗出物分割完成",
                            "指标": {
                                "硬渗出物区域占比": 0.012,
                                "硬渗出物分布指数": 1.809,
                                "硬渗出物平均面积": 615.25
                            }
                        },
                        "软渗出物分割": {
                            "状态": "软渗出物分割完成",
                            "指标": {
                                "软渗出物区域占比": 0.003,
                                "软渗出物分布指数": 0.5,
                                "软渗出物平均面积": 120.0
                            }
                        }
                    },
                    "量化分析指标": {
                        "微动脉瘤数量": 8,
                        "微动脉瘤区域占比": 0.005,
                        "出血区域占比": 0.023,
                        "出血区域总大小": 4045.0,
                        "硬渗出物区域占比": 0.012,
                        "硬渗出物分布指数": 1.809,
                        "硬渗出物平均面积": 615.25,
                        "软渗出物区域占比": 0.003,
                        "软渗出物分布指数": 0.5,
                        "软渗出物平均面积": 120.0
                    }
                }
            },
            "quality_assess": {
                "status": "success",
                "result": {
                    "quality_score": 0.85,
                    "assessment": "Good Quality"
                }
            }
        }

        if tool_name in mock_results:
            logger.info(f"使用预定义模拟结果: {tool_name}")
            return mock_results[tool_name]
        else:
            return {
                "status": "success",
                "result": {"mock": True, "tool": tool_name}
            }

    def clear_cache(self):
        """清除缓存"""
        self.tool_cache = {}
        cache_file = self.cache_dir / "tool_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("缓存已清除")

    def clear_tool_cache(self, tool_name: str):
        """清除特定工具的缓存
        
        Args:
            tool_name: 要清除缓存的工具名称
        """
        keys_to_remove = []
        for key in self.tool_cache:
            if key.startswith(f"{tool_name}:"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.tool_cache[key]
            logger.info(f"已清除工具缓存: {tool_name} (key: {key[:32]}...)")
        
        if keys_to_remove:
            # 保存更新后的缓存
            cache_file = self.cache_dir / "tool_cache.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.tool_cache, f)
                logger.info(f"已更新缓存文件，移除了 {len(keys_to_remove)} 条记录")
            except Exception as e:
                logger.warning(f"保存缓存失败: {e}")
        else:
            logger.info(f"未找到工具 {tool_name} 的缓存记录")

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            "total_entries": len(self.tool_cache),
            "cache_dir": str(self.cache_dir),
            "cache_enabled": self.enable_cache
        }


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 创建执行器
    executor = RealToolExecutor(
        tools_interface_path="../agents/fundus_tools_agent",
        enable_cache=True
    )

    
