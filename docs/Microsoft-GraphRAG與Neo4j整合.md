# Microsoft GraphRAG 與 Neo4j 整合

## 結論

Microsoft GraphRAG 可以與 Neo4j 整合，但目前沒有把 Neo4j 當成原生圖譜儲存後端。Microsoft GraphRAG 預設將索引結果輸出為 Parquet，Embedding 則寫入設定的向量資料庫；若要使用 Neo4j，需要增加資料轉換、匯入或自訂查詢層。

對本專案而言，較合適的策略是繼續以 Neo4j 作為主要圖譜資料庫，保留既有 PDF 路由、Hybrid Search、圖譜擴展、Reranker 與評測流程，再選擇性引入 Microsoft GraphRAG 的 Leiden 社群偵測和 Community Reports。

## Microsoft GraphRAG 的預設輸出

```text
Microsoft GraphRAG Indexer
    ├── entities.parquet
    ├── relationships.parquet
    ├── communities.parquet
    ├── community_reports.parquet
    └── embeddings → 向量資料庫（預設 LanceDB）
```

標準 indexing pipeline 會抽取實體、關係與 claims，執行社群偵測，建立多層級 Community Reports，並產生所需 Embedding。索引資料預設保存為 Parquet，而不是直接寫入圖資料庫。

參考：[Microsoft GraphRAG Indexing Overview](https://microsoft.github.io/graphrag/index/overview/)

## 整合方案

### 方案一：Microsoft GraphRAG 建立索引後匯入 Neo4j

```text
PDF
 ↓
Microsoft GraphRAG Indexer
 ↓
Parquet 輸出
 ↓
資料轉換／匯入
 ↓
Neo4j
```

可以匯入 Neo4j 的資料包括：

- Documents
- Text Units
- Entities
- Relationships
- Communities
- Community Reports
- Embeddings
- 各資料與原始文字之間的來源關聯

優點：

- 保留 Microsoft 的 Leiden communities 與 Community Reports。
- 可使用 Neo4j Browser、Cypher、向量索引與圖遍歷。
- 能在 Neo4j 上重新實作 Local Search 與 Global Search。

缺點：

- 必須維護 Parquet 到 Neo4j 的欄位映射與匯入程式。
- Microsoft GraphRAG 更新輸出格式時，轉換程式可能需要同步修改。
- 社群提供的舊匯入範例可能不相容於新版輸出格式。

參考：

- [Neo4j：Integrating Microsoft GraphRAG into Neo4j](https://neo4j.com/blog/developer/microsoft-graphrag-neo4j/)
- [Microsoft GraphRAG：Neo4j import notebook 相容性議題](https://github.com/microsoft/graphrag/issues/1550)

### 方案二：保留既有建圖，使用 Microsoft BYOG

Microsoft GraphRAG 支援 Bring Your Own Graph。可以把 Neo4j 內的既有資料轉換為：

- `entities.parquet`
- `relationships.parquet`
- `text_units.parquet`

然後只執行社群分析、報告生成與必要的 Embedding workflow：

```yaml
workflows:
  - create_communities
  - create_community_reports
  - generate_text_embeddings
```

主要資料要求：

- Entity：`id`、`title`、`description`、`text_unit_ids`
- Relationship：`source`、`target`、`description`、`weight`、`text_unit_ids`
- Text Unit：原始 chunk 文字及其與 entity、relationship 的關聯
- Relationship 的 `weight` 會影響 Leiden community detection
- Local、DRIFT 與 Basic Search 還需要 Text Units 和 Embeddings

優點：

- 保留本專案現有的自訂 Schema、來源頁碼與 Neo4j 資料模型。
- 不必重做已完成的 PDF 解析、實體關係抽取與檢索流程。
- 可專注引入目前缺少的 Community Detection 與 Community Reports。

參考：[Microsoft GraphRAG Bring Your Own Graph](https://github.com/microsoft/graphrag/blob/main/docs/index/byog.md)

### 方案三：自訂 Microsoft GraphRAG 的儲存與查詢層

可以自行開發：

- Neo4j output adapter
- Neo4j vector-store adapter
- Neo4j Local Search context builder
- Neo4j Global Search community loader

這種方式整合程度最高，但開發與維護成本也最高。目前不能只靠類似下列設定直接完成：

```yaml
graph_store:
  type: neo4j
```

官方預設並沒有這個 Neo4j graph-store 選項，仍需要自訂轉換或整合程式。

參考：[Microsoft GraphRAG：Knowledge Graph Storage Discussion](https://github.com/microsoft/graphrag/discussions/328)

## 本專案建議架構

本專案目前已具備：

- PDF chunk 與頁碼追蹤
- 可編輯的領域 Schema
- Entity／Relationship 抽取
- Neo4j 儲存
- 向量與全文 Hybrid Search
- 圖譜鄰居擴展
- 可選 Reranker
- PDF 路由
- Recall@5／MRR 評測

因此不建議直接用 Microsoft GraphRAG 取代整個流程。建議採用以下方式：

```text
既有 PDF 建圖流程
  ↓
Neo4j Entity／Relationship／Chunk
  ↓
匯出 Microsoft BYOG 格式
  ↓
Microsoft Leiden Community Detection
  ↓
產生 Community Reports
  ↓
將 Community 與 Report 寫回 Neo4j
  ↓
新增全域搜尋模式
```

### 建議的 Neo4j 資料模型

```text
(:Entity)-[:IN_COMMUNITY]->(:Community)
(:Community)-[:PARENT_OF]->(:Community)
(:Community)-[:HAS_REPORT]->(:CommunityReport)
(:CommunityReport)-[:SUPPORTED_BY]->(:Chunk)
(:Chunk)-[:FROM_DOCUMENT]->(:Document)
```

所有 Community Reports 仍應保留來源 PDF、頁碼與 chunk，避免全域摘要失去可追溯性。

## 建議的查詢模式

整合完成後可提供四種模式：

| 模式 | 策略 | 適用問題 |
|---|---|---|
| 基本檢索 | Hybrid Search | 錯誤碼、型號、精確操作步驟 |
| 關聯擴展檢索 | Hybrid Search＋Neo4j 鄰居擴展 | 實體關係、原因與影響 |
| 全域搜尋 | Community Reports＋Map-Reduce | 全部文件的主題、趨勢與共同問題 |
| 探索搜尋 | Community Reports＋Local Search | 從概括問題逐步探索細節，類似 DRIFT |

## 推薦實作順序

1. 定義現有 Neo4j 節點與 Microsoft BYOG 欄位的映射。
2. 匯出 entities、relationships 和 text units。
3. 執行 `create_communities` 與 `create_community_reports`。
4. 將 communities 和 reports 寫回 Neo4j。
5. 保存 Community Report 到來源 chunk／頁碼的對應。
6. 新增「全域搜尋」模式與 Map-Reduce 回答流程。
7. 建立全域問題測試集，與既有基本檢索及關聯擴展檢索比較。

## 最終建議

最適合本專案的方式是採用方案二：Neo4j 繼續作為主要圖譜資料庫，Microsoft GraphRAG 僅負責社群偵測與 Community Reports。這樣能保留既有技術手冊精確檢索、PDF 路由與來源追蹤能力，同時補上跨文件全域分析能力。
