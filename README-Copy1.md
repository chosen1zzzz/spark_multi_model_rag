
# Spark-Multi-modal RAG 图文问答项目

## 快速开始：克隆仓库

你可以通过以下命令克隆本项目源码：

```sh
git clone https://github.com/li-xiu-qi/spark_multi_rag.git
```

克隆后请进入项目目录，按下文说明配置和运行。

> 本项目主要在本地A6000显卡环境下运行，核心依赖本地部署的 qwen3（文本生成）和 bge-m3（文本向量化）模型。

本项目旨在实现一个多模态 RAG（Retrieval-Augmented Generation）图文问答系统，支持从 PDF 文档自动解析、内容结构化、分块、向量化检索到大模型生成式问答的全流程。

## 主要文件说明

| 文件名 | 作用简介 |
| ------ | ------------------------------------------------------------ |
| fitz_parse_pdf.py | 一键式数据处理主脚本，自动完成 PDF 解析、内容结构化、分块合并 |
| rag_from_page_chunks.py | RAG 检索与问答主脚本，支持向量检索与大模型生成式问答 |
| get_text_embedding.py | 文本向量化，支持批量处理 |
| extract_json_array.py | 从大模型输出中提取 JSON 结构，保证结果可解析 |
| all_pdf_page_chunks.json | 所有 PDF 分页内容的合并结果，供 RAG 检索 |
| .env | 环境变量配置文件，存放 API 密钥、模型等参数 |
| caches/ | 预留目录（可选） |
| datas/ | 原始 PDF 文件及测试集存放目录 |
| data_base_json_content/ | PDF 解析后的内容列表（中间结果） |
| data_base_json_page_content/ | 分页后的内容（中间结果） |
| output/ | 其他输出文件目录 |

## 其他PDF解析方案：fitz（PyMuPDF）方案

本项目也支持使用 `fitz`（PyMuPDF）库进行PDF文本解析，作为MinerU之外的另一种可选方案。

- 相关脚本：`fitz_pipeline_all.py`
- 依赖安装：

  ```sh
  pip install pymupdf
  ```

- 用法说明：
  运行 `fitz_pipeline_all.py` 可直接遍历 `datas/` 目录下所有PDF文件，按页提取文本并生成 `all_pdf_page_chunks.json`，适用于纯文本型PDF的快速解析。

  ```sh
  python fitz_pipeline_all.py
  ```

- 适用场景：
  - 仅需提取PDF文本内容、无需复杂版面结构还原时可选用此方案。
  - 适合大批量、纯文本PDF的快速处理。
  - 若需表格/图片/复杂结构还原，建议优先使用MinerU方案。

> 注：fitz方案无需依赖GPU，适合轻量级场景。

## 数据准备

### datas目录内容说明

`datas/` 目录包含项目运行所需的原始数据文件：

- **比赛数据文件**：
  - `多模态RAG图文问答挑战赛训练集.json` - 用于模型训练的数据集
  - `多模态RAG图文问答挑战赛测试集.json` - 用于模型测试的数据集  
  - `多模态RAG图文问答挑战赛提交示例.json` - 提交格式示例

- **财报数据库文件夹**：
  - 包含大量上市公司研究报告和财务报告PDF文件
  - 主要覆盖：伊利股份、广联达、千味央厨等公司的深度研究报告
  - 包含各公司年度报告、季度报告等财务文档

### datas目录结构

```
datas/
├── 多模态RAG图文问答挑战赛训练集.json
├── 多模态RAG图文问答挑战赛测试集.json
├── 多模态RAG图文问答挑战赛提交示例.json
└── 财报数据库/
    ├── 伊利股份相关研究报告/
    │   ├── 伊利股份-公司研究报告-平台化的乳企龙头引领行业高质量转型-25071638页.pdf
    │   ├── 伊利股份内蒙古伊利实业集团股份有限公司2024年年度报告.pdf
    │   └── ... (其他伊利股份研究报告)
    ├── 广联达公司研究报告/
    │   ├── 广联达-公司深度报告-数字建筑全生命周期解决方案领军者-24041638页.pdf
    │   ├── 广联达-云计算稀缺龙头迎收入利润率双升-21080125页.pdf
    │   └── ... (其他广联达研究报告)
    ├── 千味央厨公司研究报告/
    │   ├── 千味央厨-公司深度报告-深耕B端蓝海扬帆-21090629页.pdf
    │   ├── 千味央厨-公司研究报告-大小B双轮驱动餐饮市场大有可为-23061240页.pdf
    │   └── ... (其他千味央厨研究报告)
    └── 其他上市公司研究/
        ├── 中恒电气-公司研究报告-HVDC方案领头羊AI浪潮下迎新机-25071124页.pdf
        ├── 亚翔集成-公司研究报告-迎接海外业务重估-25071324页.pdf
        ├── 传音控股-公司研究报告-非洲手机领军者多元化布局品类扩张生态链延伸打开成长空间-25071636页.pdf
        └── ... (其他公司研究报告)
```

> **重要提示**：使用本项目前，请确保 `datas/` 目录中包含所需的数据文件。如果没有数据，可以从以下地址下载：
>
> <https://challenge.xfyun.cn/topic/info?type=Multimodal-RAG-QA&option=stsj&ch=dwsf2517>
>
> 下载后请将数据文件复制到 `datas/` 目录中。

## 目录结构说明

### 目录结构

```
multi_rag/
├── all_pdf_page_chunks.json                # 所有PDF分页内容合并结果，供RAG检索
├── caches/                                # 缓存目录（如sqlite等，含cache.db等）
├── data_base_json_content/                # 每个PDF文档解析后的内容（结构化，按文档分文件夹）
├── data_base_json_page_content/           # 每个PDF分页内容（按文档分文件夹）
├── datas/                                 # 原始PDF及测试集
│   ├── 多模态RAG图文问答挑战赛训练集.json
│   ├── 多模态RAG图文问答挑战赛测试集.json
│   ├── 多模态RAG图文问答挑战赛提交示例.json
│   └── 财报数据库/                        # 各公司研究报告PDF子目录
├── extract_json_array.py                  # 从大模型输出中提取JSON结构的工具
├── fitz_pipeline_all.py                   # fitz(Pymupdf)方案PDF解析脚本（纯文本PDF快速处理）
├── get_text_embedding.py                  # 文本向量化与批量embedding脚本
├── image_utils/                           # 图像处理与分析相关辅助脚本
├── main.py                                # 项目主入口脚本
├── mineru_parse_pdf.py                    # MinerU PDF解析工具（复杂结构还原）
├── mineru_pipeline_all.py                 # MinerU一键式处理脚本
├── output/                                # 输出目录（如评测、推理、测试结果等）
│   └── test/
├── pyproject.toml                         # Python项目配置文件
├── rag_from_page_chunks.py                # RAG检索与问答主脚本（支持批量评测/交互问答）
├── rag_top1_pred.json                     # RAG预测结果（结构化）
├── rag_top1_pred_raw.json                 # RAG原始预测结果
├── __pycache__/                           # Python缓存目录
└── README.md                              # 项目说明文档
```

- `data_base_json_content/` 和 `data_base_json_page_content/` 下为每个PDF文档单独的解析/分页结果文件夹，便于溯源和增量处理。
- `output/` 目录下可包含评测、推理、测试等各类输出结果。
- `image_utils/` 目录为图片分析与处理相关的辅助脚本。
- `caches/`、`__pycache__/` 等为缓存和Python自动生成目录。

> 所有自动生成的json文件和中间目录已加入.gitignore，默认不会被上传到git。

> 说明：所有自动生成的json文件和中间目录已加入.gitignore，默认不会被上传到git。

- `pipeline_all.py`：一键式数据处理主脚本，自动完成 PDF 解析、内容结构化、分块合并。
- `rag_from_page_chunks.py`：RAG 检索与问答主脚本，支持向量检索与大模型生成式问答。
- `get_text_embedding.py`：文本向量化与缓存。
- `extract_json_array.py`：从大模型输出中提取 JSON 结构。
- `all_pdf_page_chunks.json`：所有 PDF 分页内容的合并结果，供 RAG 检索。
- 其他目录：
  - `datas/`：原始 PDF 及测试集。
  - `data_base_json_content/`：PDF 解析后的内容列表。
  - `data_base_json_page_content/`：分页后的内容。
  - `caches/`、`output/` 等：缓存与输出。

## 使用流程

1. **一键PDF数据处理**

   运行 `mineru_pipeline_all.py`，自动完成：
   - 遍历 `datas/` 目录下所有PDF，解析为结构化内容（存入 `data_base_json_content/`）
   - 分页处理（存入 `data_base_json_page_content/`）
   - 合并所有分页内容为 `all_pdf_page_chunks.json`，供后续RAG检索

   ```sh
   python mineru_pipeline_all.py
   ```

   > 如需快速处理纯文本PDF，也可用 `fitz_pipeline_all.py`，无需GPU。

2. **RAG 检索与问答**

   运行 `rag_from_page_chunks.py`，加载分块内容，支持如下功能：
   - 自动读取测试集，批量评测并输出结构化结果（如 `rag_top1_pred.json`）
   - 支持自定义问题的检索与生成，交互式问答

   ```sh
   python rag_from_page_chunks.py
   ```

3. **环境变量与模型配置**

   请在 `.env` 文件中配置本地或云端API参数，示例：

   ```env
   LOCAL_API_KEY=anything
   LOCAL_BASE_URL=http://localhost:10002/v1
   LOCAL_TEXT_MODEL=qwen3
   LOCAL_EMBEDDING_MODEL=bge-m3
   # 或配置硅基流动平台API参数
   # GUIJI_API_KEY=xxx
   # GUIJI_BASE_URL=xxx
   # GUIJI_TEXT_MODEL=Qwen2.5-VL-32B-Instruct
   ```

   > 推荐A6000等高性能显卡本地部署。如无本地环境，也可用云API参数直接运行。

4. **依赖安装**

   推荐 Python 3.8+，首次运行前请安装依赖：

   ```sh
   pip install -r requirements.txt
   ```

> 以上流程涵盖了从PDF解析、内容结构化、分块、向量化检索到大模型问答的全链路。所有中间结果和输出均自动存放于对应目录，便于溯源和复用。

3. **环境变量配置**

   请在 `.env` 文件中配置以下内容（示例）：

   ```env
   # 本地推理（推荐本地部署时使用）
   LOCAL_API_KEY=anything
   LOCAL_BASE_URL=http://localhost:10002/v1
   LOCAL_TEXT_MODEL=qwen3           # 本地大模型，推荐 qwen3
   LOCAL_EMBEDDING_MODEL=bge-m3     # 本地embedding模型，推荐 bge-m3
   
   # 你也可以直接将 LOCAL_API_KEY、LOCAL_BASE_URL、LOCAL_TEXT_MODEL、LOCAL_EMBEDDING_MODEL
   # 换成硅基流动平台的API参数（如GUIJI_API_KEY、GUIJI_BASE_URL、Qwen/Qwen2.5-VL-32B-Instruct等），无需本地部署也可直接运行。
   ```

   > 推荐在A6000等高性能显卡环境下本地部署上述模型，确保推理效率。如无本地环境，也可直接用硅基流动API参数兼容运行。

## 硬件资源建议

本项目依赖的 MinerU PDF 解析工具对硬件有如下建议（详见 [MinerU官方文档](https://github.com/opendatalab/MinerU/blob/master/README_zh-CN.md)）：

| 资源类型 | 最低要求 | 推荐配置 |
| -------- | ----------------------------- | ----------------------------- |
| GPU      | Turing及以后架构，6G显存以上或Apple Silicon | Turing及以后架构，8G显存以上 |
| 内存     | 16G以上                        | 32G以上                        |

也支持CPU推理，但速度会慢一些。

## 多模态与Embedding模型API说明

本项目支持多模态大模型推理，推荐如下两种方式：

1. **硅基流动云API**
   - 多模态模型：如 Qwen/Qwen2.5-VL-32B-Instruct，API地址见 [硅基流动](https://cloud.siliconflow.cn/i/FcjKykMn)
   - 图片视觉分析能力：pipeline_all.py 支持自动为图片补全caption，默认调用硅基流动Qwen/Qwen2.5-VL-32B-Instruct模型（无需本地部署，API Key见.env的GUIJI_API_KEY）。如需关闭图片caption补全，可在代码中设置 enable_image_caption=False。
   - Embedding模型：如 BAAI/bge-m3、重排序模型 BAAI/bge-reranker-v2-m3，均可免费调用
   - 只需在 `.env` 中配置 GUIJI_API_KEY、GUIJI_BASE_URL、GUIJI_TEXT_MODEL、GUIJI_FREE_TEXT_MODEL、LOCAL_EMBEDDING_MODEL 等参数即可

2. **本地xinference统一部署**
   - 支持本地多模态模型、embedding模型、mineru等一站式推理
   - 推荐A6000等高性能显卡环境
   - 参考 [xinference官方文档](https://inference.readthedocs.io/en/latest/) 部署

> 你可以根据自身条件选择云API或本地推理，硅基流动平台和xinference均支持多模态和embedding模型。

## 本地模型部署推荐

本项目推荐使用 [xinference](https://inference.readthedocs.io/en/latest/) 进行本地大模型和embedding模型的统一部署与管理。

请参考官方文档完成 xinference 的安装与模型加载，确保 `LOCAL_BASE_URL` 指向你的 xinference 服务地址。

## 依赖安装

推荐使用 Python 3.8+，安装依赖：

```sh
pip install -r requirements.txt
```

## 主要特性

- 支持批量 PDF 自动解析与内容结构化
- 支持分页内容分块与向量化检索
- 支持大模型生成式问答，输出结构化 JSON
- 支持多线程批量评测

## 适用场景

- 金融、法律、科研等领域的 PDF 文档智能问答
- 多文档、多页内容的高效检索与分析
