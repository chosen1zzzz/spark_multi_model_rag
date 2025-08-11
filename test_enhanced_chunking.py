"""
测试增强分块效果的脚本
"""
import json
from pathlib import Path
from enhanced_mineru_pipeline import process_mineru_to_enhanced_chunks, compare_chunking_strategies


def test_sample_content():
    """测试样本内容的分块效果"""
    print("=== 测试样本内容分块 ===")
    
    # 创建测试样本
    sample_content = """
# 1. 公司介绍：聚焦数字能源产业链，软、硬件协同发展

# 1.1 中恒电气：专注零碳智能社会建设的数字能源公司

杭州中恒电气股份有限公司（简称"中恒电气"）创立于 1996 年，2010 年于深交所挂牌上市。公司专注于成为零碳智能社会建设的数字能源公司，始终坚持主业发展，持续打造涵盖绿色节能设备、能源数字化软件、能源云服务平台"三位一体"的数字能源产业体系。

<table><tr><td>市场数据：2025年7月10日</td></tr><tr><td>收盘价（元)：</td><td>15.12</td></tr><tr><td>总股本(亿股)：</td><td>5.64</td></tr><tr><td>流通股本 (亿股)：</td><td>5.58</td></tr><tr><td>流通市值 (亿元)：</td><td>84.39</td></tr></table>

![图 1：公司发展历程](images/df00fda3a61c00abfdce8e5e3df74cf4414ef3e5a9ed0553b43faf6efc56481e.jpg)

公司聚焦绿色ICT 基础设施、新型电力系统及综合能源服务等领域，主要业务包含数据中心与站点能源、数字电网与综合能源服务、电力电源、新能源车充换电等。

# 1.2 股权结构稳定，管理层经验丰富

公司股权结构稳定，朱国锭先生为公司实控人，包晓茹女士为公司董事长，二者系夫妻关系，为一致行动人。截至 2024 年底，朱国锭先生和包晓茹女士合计持有公司 $4 1 . 4 \\%$ 的股权。
"""
    
    from mineru_advanced_chunker import MinerUAdvancedChunker
    
    chunker = MinerUAdvancedChunker(chunk_size=300, overlap_size=30)
    metadata = {"page": "0", "file_name": "test.pdf"}
    
    chunks = chunker.chunk_mineru_content(sample_content, metadata)
    
    print(f"样本内容长度: {len(sample_content)} 字符")
    print(f"生成chunks数量: {len(chunks)}")
    print("\n各chunk详情:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({chunk['type']}):")
        print(f"  ID: {chunk['id']}")
        print(f"  长度: {len(chunk['content'])} 字符")
        print(f"  内容预览: {chunk['content'][:100]}...")


def test_retrieval_simulation():
    """模拟检索测试"""
    print("\n=== 模拟检索测试 ===")
    
    # 检查增强分块文件是否存在
    enhanced_file = "all_pdf_enhanced_chunks.json"
    if not Path(enhanced_file).exists():
        print(f"增强分块文件 {enhanced_file} 不存在，请先运行增强分块处理")
        return
    
    # 加载增强分块数据
    with open(enhanced_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # 模拟查询
    test_queries = [
        "中恒电气的营业收入",
        "HVDC技术优势",
        "公司股权结构",
        "巴拿马电源方案",
        "财务数据表格"
    ]
    
    print(f"加载了 {len(chunks)} 个chunks")
    print("模拟查询测试:")
    
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 简单的关键词匹配模拟
        relevant_chunks = []
        for chunk in chunks:
            content = chunk['content'].lower()
            if any(keyword.lower() in content for keyword in query.split()):
                relevant_chunks.append(chunk)
        
        print(f"  匹配到 {len(relevant_chunks)} 个相关chunks")
        
        # 显示最相关的chunk
        if relevant_chunks:
            best_chunk = relevant_chunks[0]
            print(f"  最佳匹配类型: {best_chunk.get('type', 'unknown')}")
            print(f"  内容长度: {len(best_chunk['content'])} 字符")
            print(f"  内容预览: {best_chunk['content'][:150]}...")


def analyze_chunk_types():
    """分析chunk类型分布"""
    print("\n=== Chunk类型分析 ===")
    
    enhanced_file = "all_pdf_enhanced_chunks.json"
    if not Path(enhanced_file).exists():
        print(f"增强分块文件 {enhanced_file} 不存在")
        return
    
    with open(enhanced_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # 统计各类型chunk
    type_stats = {}
    size_by_type = {}
    
    for chunk in chunks:
        chunk_type = chunk.get('type', 'unknown')
        content_size = len(chunk['content'])
        
        if chunk_type not in type_stats:
            type_stats[chunk_type] = 0
            size_by_type[chunk_type] = []
        
        type_stats[chunk_type] += 1
        size_by_type[chunk_type].append(content_size)
    
    print("Chunk类型统计:")
    for chunk_type, count in type_stats.items():
        avg_size = sum(size_by_type[chunk_type]) / len(size_by_type[chunk_type])
        print(f"  {chunk_type}: {count} 个, 平均大小: {avg_size:.1f} 字符")
    
    # 找出表格chunks
    table_chunks = [c for c in chunks if c.get('type') == 'table']
    print(f"\n表格chunks详情 ({len(table_chunks)} 个):")
    for i, chunk in enumerate(table_chunks[:3]):  # 只显示前3个
        print(f"  表格 {i+1}: {chunk['id']}")
        print(f"    长度: {len(chunk['content'])} 字符")
        print(f"    预览: {chunk['content'][:100]}...")


def main():
    """主测试函数"""
    print("🧪 增强分块测试套件")
    print("=" * 50)
    
    # 测试1: 样本内容分块
    test_sample_content()
    
    # 测试2: 检索模拟
    test_retrieval_simulation()
    
    # 测试3: chunk类型分析
    analyze_chunk_types()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成")


if __name__ == "__main__":
    main()
