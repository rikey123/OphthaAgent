#!/bin/bash
# CoT生成器 V2 快速启动脚本

set -e

echo "=========================================="
echo "CoT生成器 V2 - 快速启动向导"
echo "=========================================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    echo "请先安装Python 3.8+"
    exit 1
fi

PYTHON=$(which python3)
echo "✓ Python: $PYTHON"

# 检查当前目录
if [ ! -f "main_v2.py" ]; then
    echo "❌ 错误: 请在generator目录下运行此脚本"
    exit 1
fi

echo "✓ 当前目录正确"

# 步骤1: 安装依赖
echo ""
echo "步骤1: 检查依赖..."
if [ ! -d "venv" ]; then
    echo "  创建虚拟环境..."
    $PYTHON -m venv venv
    source venv/bin/activate
    echo "  安装依赖包..."
    pip install --upgrade pip
    pip install -r requirements_v2.txt
    echo "✓ 依赖安装完成"
else
    echo "✓ 虚拟环境已存在"
    source venv/bin/activate
fi

# 步骤2: 配置环境变量
echo ""
echo "步骤2: 配置环境..."
if [ ! -f ".env" ]; then
    echo "  未找到.env文件，复制示例..."
    cp .env.example .env
    echo "⚠️  请编辑.env文件，填入您的API密钥"
    echo "   nano .env  # 或使用其他编辑器"
    read -p "  按Enter键继续..."
else
    echo "✓ .env文件已存在"
fi

# 步骤3: 测试组件
echo ""
echo "步骤3: 测试组件..."
read -p "是否运行组件测试? (y/N): " run_test

if [[ $run_test =~ ^[Yy]$ ]]; then
    echo "  运行测试..."
    $PYTHON test_components.py
    if [ $? -eq 0 ]; then
        echo "✓ 所有测试通过"
    else
        echo "⚠️  部分测试失败，请检查错误信息"
        read -p "是否继续? (y/N): " continue_anyway
        if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "  跳过测试"
fi

# 步骤4: 选择运行模式
echo ""
echo "=========================================="
echo "请选择运行模式:"
echo "=========================================="
echo "1. 试运行（只分类，不生成）"
echo "2. 生成单个类型（如Type B纠错场景）"
echo "3. 完整生成（所有类型）"
echo "4. 自定义参数"
echo "0. 退出"
echo ""
read -p "请选择 [0-4]: " mode

case $mode in
    1)
        echo ""
        read -p "请输入数据集路径: " dataset
        echo "运行试运行模式..."
        $PYTHON main_v2.py \
            --dataset "$dataset" \
            --output /dev/null \
            --dry-run
        ;;
    2)
        echo ""
        read -p "请输入数据集路径: " dataset
        read -p "请输入输出路径: " output
        echo "请选择类型:"
        echo "  A - 标准SOP"
        echo "  B - 纠错鉴别"
        echo "  C - 级联推理"
        echo "  D - 多病并发"
        echo "  E - 拒诊异常"
        read -p "类型 [A-E]: " type_filter

        echo "运行类型过滤生成..."
        $PYTHON main_v2.py \
            --dataset "$dataset" \
            --output "$output" \
            --type-filter "$type_filter" \
            --max-workers 3 \
            --enable-cache
        ;;
    3)
        echo ""
        read -p "请输入数据集路径: " dataset
        read -p "请输入输出路径: " output
        read -p "最大并发数 (默认3): " workers
        workers=${workers:-3}

        echo "运行完整生成..."
        $PYTHON main_v2.py \
            --dataset "$dataset" \
            --output "$output" \
            --max-workers "$workers" \
            --enable-cache
        ;;
    4)
        echo ""
        echo "自定义参数模式"
        echo "示例: python main_v2.py --dataset data.jsonl --output results.json --type-filter B"
        echo ""
        read -p "请输入完整命令参数: " custom_args
        $PYTHON main_v2.py $custom_args
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

# 完成
echo ""
echo "=========================================="
echo "✓ 完成！"
echo "=========================================="

if [ -f "$output" ] && [ "$mode" != "1" ]; then
    echo ""
    echo "输出文件: $output"

    read -p "是否查看统计报告? (y/N): " view_report
    if [[ $view_report =~ ^[Yy]$ ]]; then
        report="${output%.json}_report.json"
        if [ -f "$report" ]; then
            cat "$report"
        fi
    fi

    read -p "是否运行质量验证? (y/N): " run_validation
    if [[ $run_validation =~ ^[Yy]$ ]]; then
        echo "运行验证..."
        $PYTHON trajectory_validator.py "$output"
    fi
fi

echo ""
echo "感谢使用CoT生成器 V2!"
