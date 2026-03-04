#!/usr/bin/env python3
"""
工具集成验证脚本
验证tool_call_test.py中的所有工具都已集成到real_tool_executor.py中
"""
import sys
import os
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_integration():
    """验证所有工具是否已集成"""

    print("="*70)
    print("工具集成验证 - CoT生成器 V2")
    print("="*70)

    # tool_call_test.py中使用的所有工具（已移除废弃的AutoMorph工具）
    required_tools = [
        # From tools_interface1.py
        'enhance_fundus_image',
        'fundus_lesion_segmentation',

        # From tools_interface.py
        'fov_od_localization_by_fit',
        'dme_risk_assessment',

        # From tool_interface3.py
        'segment_by_ddcs',
        'DR_Grading',
        'segment_by_AutoMorphalyzer',  # 替代老版AutoMorph

        # From tool_interface4.py
        'AMD_predict_fundus_by_deepseenet',
        'Lesion_predict_OCT_by_opticnet',
    ]

    print(f"\n需要验证的工具数量: {len(required_tools)}")
    print("\n⚠️  注意: 老版AutoMorph工具 (autoMorphProcess, run_automorph_m0_m1, etc.) 已废弃")
    print("   请使用 segment_by_AutoMorphalyzer 替代")
    print("-"*70)

    # 加载RealToolExecutor
    try:
        from generator.real_tool_executor import RealToolExecutor

        executor = RealToolExecutor(
            tools_interface_path=str(project_root / "agents" / "fundus_tools_agent"),
            enable_cache=False
        )

        print(f"\n✓ RealToolExecutor初始化成功")
        print(f"✓ 已加载工具总数: {len(executor.tools_dict)}")

    except Exception as e:
        print(f"\n✗ RealToolExecutor初始化失败: {e}")
        return False

    # 检查每个工具
    print(f"\n{'工具名称':<45} {'状态':<10} {'映射名称':<30}")
    print("-"*70)

    missing_tools = []
    available_tools = []

    for tool in required_tools:
        # 检查是否在tools_dict中
        if tool in executor.tools_dict:
            status = "✓ 可用"
            available_tools.append(tool)
            mapped_name = tool
        # 检查是否有别名映射
        elif tool in executor.TOOL_MAPPING:
            real_name = executor.TOOL_MAPPING[tool]
            if real_name in executor.tools_dict:
                status = "✓ 可用"
                available_tools.append(tool)
                mapped_name = real_name
            else:
                status = "✗ 缺失"
                missing_tools.append(tool)
                mapped_name = f"映射到{real_name}但未加载"
        else:
            status = "✗ 缺失"
            missing_tools.append(tool)
            mapped_name = "无映射"

        print(f"{tool:<45} {status:<10} {mapped_name:<30}")

    # 统计报告
    print("\n" + "="*70)
    print("验证结果")
    print("="*70)

    total = len(required_tools)
    available_count = len(available_tools)
    missing_count = len(missing_tools)

    print(f"\n总工具数: {total}")
    print(f"✓ 可用: {available_count} ({available_count/total*100:.1f}%)")
    print(f"✗ 缺失: {missing_count} ({missing_count/total*100:.1f}%)")

    if missing_tools:
        print(f"\n缺失的工具:")
        for tool in missing_tools:
            print(f"  - {tool}")
        print("\n⚠️  警告: 存在缺失工具,请检查集成!")
        return False
    else:
        print(f"\n✓ 所有工具均已成功集成!")

        # 显示工具别名映射
        print("\n" + "="*70)
        print("工具别名映射 (同一工具的不同名称)")
        print("="*70)

        # 分组显示别名
        tool_groups = {}
        for alias, real_name in executor.TOOL_MAPPING.items():
            if real_name not in tool_groups:
                tool_groups[real_name] = []
            if alias != real_name:  # 只显示别名
                tool_groups[real_name].append(alias)

        for real_name, aliases in sorted(tool_groups.items()):
            if aliases:
                print(f"\n{real_name}:")
                for alias in aliases:
                    print(f"  ↳ {alias}")

        # 显示所有可用工具
        print("\n" + "="*70)
        print("所有可用工具列表 (共{}个)".format(len(executor.tools_dict)))
        print("="*70)

        tools_by_category = {
            'DR相关': [],
            'AMD相关': [],
            '青光眼相关': [],
            'OCT相关': [],
            'DME相关': [],
            '图像处理': [],
            '解剖定位': [],
            '其他': []
        }

        dr_keywords = ['dr', 'grading', 'lesion', 'ddcs']
        amd_keywords = ['amd']
        glaucoma_keywords = ['automorph', 'morph', 'quantitative']
        oct_keywords = ['oct', 'opticnet']
        dme_keywords = ['dme']
        image_keywords = ['enhance', 'quality', 'crop']
        anatomy_keywords = ['fovea', 'od', 'vessel', 'fov', 'localization']

        for tool in sorted(executor.tools_dict.keys()):
            tool_lower = tool.lower()
            categorized = False

            for keyword in dr_keywords:
                if keyword in tool_lower:
                    tools_by_category['DR相关'].append(tool)
                    categorized = True
                    break

            if not categorized:
                for keyword in amd_keywords:
                    if keyword in tool_lower:
                        tools_by_category['AMD相关'].append(tool)
                        categorized = True
                        break

            if not categorized:
                for keyword in glaucoma_keywords:
                    if keyword in tool_lower:
                        tools_by_category['青光眼相关'].append(tool)
                        categorized = True
                        break

            if not categorized:
                for keyword in oct_keywords:
                    if keyword in tool_lower:
                        tools_by_category['OCT相关'].append(tool)
                        categorized = True
                        break

            if not categorized:
                for keyword in dme_keywords:
                    if keyword in tool_lower:
                        tools_by_category['DME相关'].append(tool)
                        categorized = True
                        break

            if not categorized:
                for keyword in image_keywords:
                    if keyword in tool_lower:
                        tools_by_category['图像处理'].append(tool)
                        categorized = True
                        break

            if not categorized:
                for keyword in anatomy_keywords:
                    if keyword in tool_lower:
                        tools_by_category['解剖定位'].append(tool)
                        categorized = True
                        break

            if not categorized:
                tools_by_category['其他'].append(tool)

        for category, tools in tools_by_category.items():
            if tools:
                print(f"\n{category} ({len(tools)}个):")
                for tool in tools:
                    print(f"  • {tool}")

        return True


if __name__ == "__main__":
    success = verify_integration()
    print("\n" + "="*70)

    if success:
        print("✓ 验证通过 - 所有工具已成功集成!")
        print("="*70)
        sys.exit(0)
    else:
        print("✗ 验证失败 - 请检查缺失的工具!")
        print("="*70)
        sys.exit(1)
