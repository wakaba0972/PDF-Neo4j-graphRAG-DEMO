# PDF Neo4j GraphRAG Demo

這是一套將 PDF 文件轉換為 Neo4j 知識圖譜，並透過混合式 GraphRAG 進行問答的本機 Web 工具。介面使用 Gradio，支援 OpenAI 相容的模型服務，可自行設定建圖、Embedding 與回答模型。

> 本專案目前定位為單機、單使用者 Demo，不會建立 Gradio 公開分享網址。

## 目錄

- [專案介紹](#專案介紹)
- [實作原理](#實作原理)
- [基本使用流程](#基本使用流程)
- [Docker：使用 Dockerfile 建立執行環境](#docker使用-dockerfile-建立執行環境)
- [部署後如何啟動](#部署後如何啟動)
- [常見問題](#常見問題)
- [非 Docker 啟動方式](#非-docker-啟動方式)
- [測試](#測試)
- [限制與注意事項](#限制與注意事項)

## 專案介紹

本專案提供以下功能：

- 建立與載入獨立專案工作區，保存連線、API、模型、處理參數、Chunk、文件、建圖狀態與問答紀錄。
- 上傳與預覽含文字層的 PDF。
- 勾選一或多份 PDF，從所選文件全文規劃實體、關係 Schema；預設選取全部 PDF。
- 以多請求並行方式抽取知識圖譜，整批完成後再統一去重整合。
- 將文件片段、實體與關係建立 Embedding，寫入 Neo4j 並建立向量與全文索引。
- 每次匯入前自動清空本工具建立的既有圖譜，再寫入本次抽取結果。
- 在獨立的 PDF 摘要頁建立路由摘要；文件識別資訊只保留主要對象及系列名稱（最多 5 項），排除出版商、文件編號、平台與附帶名稱。
- 從每份 PDF 自動建立指定數量的問題、標準答案、來源頁碼與來源 chunk；可選擇跨 PDF 並行生題，並支援手動編輯、自動保存、JSON／CSV 匯入與 JSON 匯出，再一鍵執行 RAG 回答及模型判分。
- 使用 Neo4j 官方 HybridCypherRetriever 執行向量與全文混合搜尋，可選擇以本機 Reranker 重排擴大召回的候選，再結合圖譜擴展及原文片段組成 GraphRAG 問答內容。
- 模型服務可切換 OpenAI／Ollama，Embedding 服務可切換 OpenAI／Ollama／Voyage，並保留各自的連線設定。後續模型選單會顯示已驗證的 OpenAI、Ollama 與 Voyage 模型，執行時自動使用該模型所屬服務。
- 使用 Ollama LLM 時，所有 Chat Completions 請求會自動啟用內部串流收集；系統持續接收 SSE 片段，完成後再交給既有 JSON 驗證或回答流程。未完整結束的串流會丟棄並整次重試，OpenAI 與 Embedding 請求維持非串流。
- 介面中的摘要、Schema 規劃、知識圖譜抽取與自動測試最大並行請求數均會提示：使用 Ollama 時建議設為 1，以降低小模型同時處理多個生成請求造成逾時或輸出品質下降的機率。

## 實作原理

### 建圖流程

1. 使用 PyMuPDF 讀取 PDF 文字，並排除每頁頂部與底部約 8% 的常見頁首頁尾區域。
2. 將文字切成可重疊的片段；預設片段大小為 1500 字元、重疊 200 字元，並保留頁碼。
3. 將文件分批送往建圖模型規劃 Schema。各批可同時執行，全部完成後再分層整合。
4. 依確認後的 Schema 並行抽取實體與關係，等待整批完成後統一正規化、去重及整合。
5. 分批建立原文片段、實體與關係的向量，寫入 Neo4j，並建立向量索引。

### 問答流程

送出問題時只會對「問題」建立 Embedding，不會重新計算整個圖譜的向量。系統會：

1. 使用問題向量搜尋相關原文、實體與關係。
2. 使用問題文字從 CJK 全文索引搜尋精確詞彙、錯誤碼與實體名稱。
3. 由 Neo4j 官方 HybridCypherRetriever 使用內建 naive ranker 正規化並融合兩組結果；啟用 Reranker 時召回 Top K 的 3 倍候選（最多 50 筆），停用時直接取 Top K。
4. 若勾選「使用 Reranker」，本機 Reranker 依問題詞彙匹配、完整關鍵詞、Hybrid 分數與證據類型重排，再選出 Top K；候選內容不會因此額外傳送至外部服務。
5. GraphRAG 模式再從命中的節點向外擴展相關圖譜內容，並回查 PDF 原文片段。
6. 將問題、圖譜內容及原文證據交給回答模型生成答案。

回答模型的證據內容會套用共用的 context token 預算；超出上限時保留排名較前的完整證據，必要時截短第一筆證據並保留其來源欄位。

兩個問答模式都會使用官方 HybridCypherRetriever 進行向量與全文混合檢索，並可選擇是否使用本機 Reranker；GraphRAG 會額外執行圖譜擴展。匯入階段預先建立 Embedding、向量索引與全文索引，是後續問答能快速檢索的關鍵。

自動問答測試會顯示 Recall@5 與 MRR：Recall@5 代表前 5 筆證據是否命中題目標記的來源 chunk（舊題目沒有 chunk 時改用來源頁碼），MRR 則依第一筆命中證據的排名計算倒數排名平均值。逐題結果與這些計算所需欄位會保存至專案；重新載入專案時，系統會由已保存結果重算並還原完整摘要。

向量索引會依目前 Embedding 維度命名，例如 `graph_evidence_embedding_1536`。由於 Neo4j 不允許相同標籤與屬性同時存在不同維度的向量索引，每次匯入會先刪除本工具的舊向量索引，再建立目前維度的唯一索引；不會刪除其他應用程式的索引。更換 Embedding 模型後需重新執行「Embedding 並匯入 Neo4j」。

從舊版升級時，請在介面重新執行一次「Embedding 並匯入 Neo4j」，以建立 `graph_evidence_fulltext` 全文索引及新的維度專屬向量索引；既有資料不會只因更新程式碼而自動建立索引。

## 基本使用流程

1. 在「0. 專案設定」建立新專案，或載入既有專案；完成後即可使用「2. PDF 與參數」。第 3–7 頁須先完成 Neo4j、模型與 Embedding 服務連線。刪除專案時瀏覽器會再次要求確認，確認後才永久刪除該專案的設定、文件與紀錄。
2. 在「連線設定」填入 Neo4j 與 OpenAI 相容模型服務，分別執行連線測試。
3. 在 PDF 頁上傳文件並確認解析結果。
4. 前往「3. PDF 摘要」，選擇摘要模型並建立所有 PDF 的路由摘要，確認每份摘要內容。
5. 前往「4. 建圖」，選擇全文或隨機 N 頁，執行「分析文件並規劃 Schema」。
6. 視需要修改 Schema，設定 LLM 與最大並行數，再執行「確認 Schema 並抽取知識圖譜」。
7. 選擇資料處理方式，按下「Embedding 並匯入 Neo4j」。
8. 前往「5. 自動問答測試」，指定每份 PDF 的題數及是否允許跨 PDF 並行後建立測試集。題目與答案可直接修改並自動保存，也可匯入 JSON／CSV 或匯出 JSON；可選擇是否使用 Reranker。完成後會顯示答對題數、路由與各 PDF 統計、Recall@5 及 MRR，重新載入專案仍會還原既有結果摘要。
9. 「6. 問答測試」可進行單題提問；成功結果會自動加入「7. 歷史紀錄」。
10. 所有設定、參數及處理結果會在目前專案中自動保存；每次進入「0. 專案設定」也會自動更新專案清單。

專案資料位於 `data/projects/<project-id>/`。`project.json` 保存非敏感設定、抽取出的實體與關係、Neo4j 匯入狀態、自動測試集、測試結果與問答紀錄；`documents/` 保存 PDF 副本。API endpoint、API Key 與 Neo4j 連線資料統一由 `.env` 管理，不會寫入 `project.json`。

問答頁不要求在同一工作階段先建圖；只要指定的 Neo4j 中已有本專案建立的圖譜即可使用。

## Docker：使用 Dockerfile 建立執行環境

以下步驟會使用專案根目錄的 [`Dockerfile`](Dockerfile) 建立 Python 3.12、Gradio 與 GraphRAG 相依套件完整的執行映像。主機不需要另外建立 Python 虛擬環境。

### 1. 安裝並啟動 Docker

Windows／macOS 請安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，Linux 可安裝 Docker Engine；另外需要 Git 下載原始碼。

~~~bash
git --version
docker version
~~~

`docker version` 必須同時顯示 Client 與 Server。若只有 Client，請先啟動 Docker Desktop 或 Docker Engine。

### 2. 下載原始碼

~~~bash
git clone https://github.com/wakaba0972/PDF-Neo4j-graphRAG-DEMO.git
cd PDF-Neo4j-graphRAG-DEMO
~~~

後續的 `docker build` 與 `docker run` 都要在含有 `Dockerfile`、`requirements.txt`、`src/` 和 `config/` 的專案根目錄執行。

### 3. 準備持久化設定與資料

Linux／macOS：

~~~bash
cp .env.example .env
mkdir -p data
~~~

Windows PowerShell：

~~~powershell
Copy-Item .env.example .env
New-Item -ItemType Directory -Force data
~~~

容器會掛載以下三個主機路徑，重新建立容器後資料仍會保留：

| 主機路徑 | 容器路徑 | 用途 |
|---|---|---|
| `.env` | `/app/.env` | Neo4j、OpenAI、Ollama、Voyage 的 endpoint、帳號與 API key |
| `config/model_settings.yaml` | `/app/config/model_settings.yaml` | 服務來源、模型選擇、模型白名單與 Ollama 勾選狀態 |
| `data/` | `/app/data/` | 專案 JSON、PDF 副本、測試題目與問答紀錄 |

若 Neo4j 或 Ollama 執行在 Docker 主機上，請在 `.env` 使用 `host.docker.internal`，不能使用容器自己的 `localhost`：

~~~dotenv
NEO4J_URI=bolt://host.docker.internal:7687
NEO4J_DATABASE=neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

MODEL_OLLAMA_API_BASE=http://host.docker.internal:11434/v1
MODEL_OLLAMA_API_KEY=
EMBEDDING_OLLAMA_API_BASE=http://host.docker.internal:11434/v1
EMBEDDING_OLLAMA_API_KEY=
~~~

只使用 OpenAI 或 Voyage 時，保留 `.env.example` 中對應的官方 API Base URL，並填入自己的 API key。不要將 `.env` 提交至 Git。

### 4. 使用 Dockerfile 建置映像

~~~bash
docker build --pull -t pdf-graphrag:latest .
~~~

此命令會依 Dockerfile 執行以下工作：

1. 下載 `python:3.12-slim-bookworm` 基底映像。
2. 安裝 `requirements.txt` 中的 Python 套件。
3. 複製 `src/` 與預設 `config/`。
4. 建立容器內的 `/app/data`，開放 Gradio 的 7860 port，並設定 HTTP health check。

確認映像已建立：

~~~bash
docker image ls pdf-graphrag
~~~

### 5. 建立並啟動容器

Linux／macOS：

~~~bash
docker run -d \
  --name pdf-graphrag \
  --restart unless-stopped \
  -p 7860:7860 \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/.env:/app/.env" \
  -v "$(pwd)/config/model_settings.yaml:/app/config/model_settings.yaml" \
  -v "$(pwd)/data:/app/data" \
  pdf-graphrag:latest
~~~

Windows PowerShell：

~~~powershell
docker run -d --name pdf-graphrag --restart unless-stopped -p 7860:7860 --add-host=host.docker.internal:host-gateway -v "$PWD/.env:/app/.env" -v "$PWD/config/model_settings.yaml:/app/config/model_settings.yaml" -v "$PWD/data:/app/data" pdf-graphrag:latest
~~~

參數用途：

- `-p 7860:7860`：將主機的 7860 port 對應到 Gradio。
- `--restart unless-stopped`：Docker 服務重新啟動後自動恢復容器。
- `--add-host=host.docker.internal:host-gateway`：讓 Linux 容器能連到主機上的 Ollama／Neo4j。
- 三個 `-v`：把設定與專案資料保存在主機，不隨容器刪除。

### 6. 驗證環境

~~~bash
docker ps --filter name=pdf-graphrag
docker logs -f pdf-graphrag
~~~

按 `Ctrl+C` 只會離開日誌畫面，不會停止容器。也可查看 Dockerfile 設定的健康狀態：

~~~bash
docker inspect --format '{{.State.Health.Status}}' pdf-graphrag
~~~

狀態成為 `healthy` 後，開啟 <http://localhost:7860>。若主機 7860 已被占用，可把啟動參數改成 `-p 8080:7860`，再開啟 <http://localhost:8080>。

進入介面後，在「1. 連線設定」測試 Neo4j、模型與 Embedding 服務。Ollama 請取得模型清單並勾選需要的模型；OpenAI／Voyage 則填入 API key 後測試連線。介面更新的連線資料與模型選擇會寫回已掛載的 `.env` 和 `config/model_settings.yaml`。

## 部署後如何啟動

首次執行 docker run 後，容器名稱會是 pdf-graphrag。日後不需要再次 git clone 或 docker build。

啟動既有容器：

~~~bash
docker start pdf-graphrag
~~~

查看狀態及日誌：

~~~bash
docker ps --filter name=pdf-graphrag
docker logs -f pdf-graphrag
~~~

停止服務：

~~~bash
docker stop pdf-graphrag
~~~

部署命令包含 --restart unless-stopped，Docker 服務重啟後通常會自動啟動容器；若曾手動停止，請執行 docker start pdf-graphrag。

### 更新到新版

~~~bash
git pull
docker build --pull -t pdf-graphrag:latest .
docker stop pdf-graphrag
docker rm pdf-graphrag
~~~

接著重新執行前一節的 docker run。掛載於專案 data 目錄的資料不會因移除容器而消失。

## 常見問題

### 網頁無法開啟

~~~bash
docker ps -a --filter name=pdf-graphrag
docker logs pdf-graphrag
~~~

若 7860 已被占用，將啟動參數改為 -p 8080:7860，再開啟 http://localhost:8080。

### Neo4j 或模型服務連線失敗

- 宿主機服務請使用 host.docker.internal，不要填 localhost 或 127.0.0.1。
- 確認 Neo4j Bolt 連接埠通常為 7687，且帳號、密碼、資料庫名稱正確。
- 確認本機服務已監聽容器可連線的網路介面。
- Linux 無法解析 host.docker.internal 時，確認 docker run 包含 --add-host=host.docker.internal:host-gateway。
- 修改 `.env` 或 `config/model_settings.yaml` 後可在介面重新讀取，或執行 `docker restart pdf-graphrag`。

### 設定或資料沒有保留

確認 `.env`、`config/model_settings.yaml` 與 `data` 掛載路徑存在且有寫入權限。連線設定寫回 `.env`，模型設定寫回 YAML，專案資料寫入 `data`。

## 非 Docker 啟動方式

需要 Python 3.11 以上版本。Linux 可使用一鍵腳本：

~~~bash
./start.sh
~~~

或手動啟動：

~~~bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python src/app.py
~~~

## 測試

~~~bash
. .venv/bin/activate
pytest
~~~

## 限制與注意事項

- PDF 必須包含可選取的文字層；目前不提供 OCR。
- 加密 PDF 不支援。
- 建議先以少量頁面驗證 Schema、模型輸出及 Neo4j 寫入結果，再處理大型文件。
- 每次執行「Embedding 並匯入 Neo4j」都會先刪除目前 Database 中由本工具建立的圖譜；不會刪除其他標籤的資料。
- .env 可能包含 API Key 與 Neo4j 密碼，請勿提交至 Git；專案已透過 .gitignore 排除。
- PyMuPDF 採 AGPL／商業雙授權，封裝、散布或商用前請確認授權需求。
