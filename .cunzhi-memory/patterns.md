# 常用模式和最佳实践

- 用户询问分块策略优化：从按页分块改为更细粒度的分块（段落、句子），考虑重叠分块策略处理跨块内容，提升检索精度和召回率
- 用户询问检索优化：当前Top-K检索信噪比低，建议引入重排模型，先召回20个候选块，再用强模型重排选出最相关的5个，提升检索质量
- 用户需要优化RAG系统的prompt，要求：1)在多个来源中选择最精确的信息 2)信息不足时回答"根据现有信息无法回答"而不是产生幻觉 3)需要分层验证型prompt设计
- 已完成RAG系统prompt优化：1)基础RAG(rag_from_page_chunks.py)增加了分层验证逻辑和反幻觉机制 2)增强RAG(enhanced_retrieval.py)增加了重排分数评估和多源信息整合策略 3)两个版本都强化了"根据现有信息无法回答"的判断标准
- 删除了enhanced_retrieval.py中未使用的QueryClassifier类，简化了代码结构，该类原本用于查询分类但在实际系统中被禁用，采用通用重排策略
- 已完成多模态RAG系统实现：1)generate_image_embeddings.py用于图片向量化 2)multimodal_retriever.py实现文本+图片联合检索 3)vlm_generator.py集成GLM-4.1V多模态生成 4)multimodal_rag_main.py主程序支持测试和交互模式
