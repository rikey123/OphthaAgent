from agents.rag_system.rag_system.rag_system import *
import os
os.environ["OPENAI_API_KEY"] = "<ANON_API_KEY>"
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
def rag_query(query):
    print("步骤1: 加载配置并初始化系统...")

    config = RAGConfig.from_json("./agents/rag_system/examples/config_with_ocr_table.json")
    rag = RAGSystem(config)
    rag.load_index("./agents/rag_system/examples/my_rag_index")
    # 4. 执行检索
    print("\n步骤4: 执行检索...")

    # 修改成了检索最相关的5个（11.3）
    results = rag.retrieve(query, top_k=3)

    # 5. 显示结果
    print(f"\n查询: {query}")
    print(f"找到 {len(results)} 个相关结果:\n")

    for i, result in enumerate(results, 1):
        print(f"--- 结果 {i} (相似度: {result['score']:.4f}) ---")
        print(f"文本: {result['text']}...")
        print(f"来源: {result['metadata'].get('filename', 'unknown')}\n")
    result_text = [result["text"] for result in results]
    return result_text

if __name__ == '__main__':
    rag_query("糖尿病视网膜病变如何进行诊断，眼底图像有哪些量化指标")