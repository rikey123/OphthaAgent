from rag_system.rag_system import *
os.environ["OPENAI_API_KEY"] = "<ANON_API_KEY>"
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
def build_base():
    # 1. 从配置文件创建RAG系统
    print("步骤1: 加载配置并初始化系统...")
    config_path = "./examples/config_with_ocr_table.json"
    config = RAGConfig.from_json(config_path)
    rag = RAGSystem(config)

    # 2. 构建索引（指定PDF目录）
    print("\n步骤2: 构建索引...")
    pdf_directory = "./examples/sample_pdfs"  # 修改为你的PDF目录

    if not os.path.exists(pdf_directory):
        print(f"错误: PDF目录不存在: {pdf_directory}")
        print("请创建目录并放入PDF文件，或修改pdf_directory变量")
        return

    rag.build_index(pdf_directory=pdf_directory)

    # 3. 保存索引（可选）
    print("\n步骤3: 保存索引...")
    rag.save_index("./examples/my_rag_index")

if __name__ == "__main__":
    build_base()