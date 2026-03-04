from rag_system.rag_system import *
os.environ["OPENAI_API_KEY"] = "<ANON_API_KEY>"
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
def rag_call(config_path,index_path,query):
    print("步骤1: 加载配置并初始化系统...")

    config = RAGConfig.from_json(config_path)
    rag = RAGSystem(config)
    rag.load_index(index_path)
    # 4. 执行检索
    print("\n步骤4: 执行检索...")

    results = rag.retrieve(query, top_k=20)

    # 5. 显示结果
    print(f"\n查询: {query}")
    print(f"找到 {len(results)} 个相关结果:\n")

    for i, result in enumerate(results, 1):
        print(f"--- 结果 {i} (相似度: {result['score']:.4f}) ---")
        print(f"文本: {result['text']}...")
        print(f"来源: {result['metadata'].get('filename', 'unknown')}\n")
    result_text = [result["text"] for result in results]
    return result_text

if __name__ == "__main__":
    config_path = "./examples/config_with_ocr_table.json"
    index_path = "./examples/my_rag_index"
    query = "请问通过眼底图像来判断糖尿病视网膜病变的具体指标是什么"  # 修改为你的查询
    rag_call(config_path,index_path,query)