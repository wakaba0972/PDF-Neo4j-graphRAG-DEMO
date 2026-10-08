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

- 建立與載入獨立車型專案；每個專案使用獨立 Neo4j Database，同一車型的多份 PDF 共用該圖譜。
- 在「1-0 專案設定」匯出／匯入 ZIP 專案封裝，保存本機專案資料夾內的設定、PDF、題庫、建圖資料、評測／實驗結果及其他檔案；匯入會建立新的專案識別碼與 Neo4j database 名稱。
- 上傳與預覽含文字層的 PDF。
- 勾選一或多份 PDF，從所選文件全文規劃實體、關係 Schema；預設選取全部 PDF。
- 以多請求並行方式抽取知識圖譜，整批完成後再統一去重整合。
- 將文件片段、實體與關係建立 Embedding，寫入 Neo4j 並建立向量與全文索引。
- 每次匯入前自動清空本工具建立的既有圖譜，再寫入本次抽取結果。
- 從每份 PDF 自動建立指定數量的問題、標準答案、來源頁碼與來源 chunk；每題先隨機抽一頁，並由 LLM 檢查內容是否足以形成完整問答，不足時逐步擴展前後頁面；可選擇跨 PDF 並行生題，並支援手動編輯、自動保存、JSON／CSV 匯入與 JSON 匯出，再一鍵執行 RAG 回答及模型判分。
- 提供「基本向量檢索」與「混合檢索」兩種模式：前者使用 Neo4j 官方 VectorCypherRetriever，後者使用 HybridCypherRetriever 執行向量與全文搜尋；混合檢索可選擇以本機 Reranker 重排擴大召回的候選，也可選擇是否擴展圖譜證據。
- 模型與 Embedding 僅支援 OpenAI API，相容的 OpenAI API Base URL 亦可設定。連線測試只用於診斷，不是後續操作的前置條件；實際呼叫失敗時會回報 API 錯誤。
- 「0-0 使用者」先選擇目前操作人（Jay、Christine、Swallow、Tai、Zhao）；選擇後才能進入其他頁面。「0-2 管理題目集」只匯入及管理中央題庫；「0-3 綁定題目集」設定各專案使用哪些題目集；「0-4 API Key 設定」提供共用 OpenAI API Base URL／Key，供對話、建圖及 Embedding 使用。使用者下拉選擇供操作標記與稽核使用，並非帳號登入或身分驗證。

Neo4j 專案隔離需使用支援多資料庫的 Neo4j Enterprise，並提供可在 `system` database 建立 database 的管理權限；Neo4j Community 與 Aura 不支援此種每專案獨立 database 建立方式。

## 實作原理

### 新增檢索策略

檢索策略集中放在專案根目錄的 `strategies/`，啟動 app 時會重新掃描 `specifications/*.yaml`，依每份規格載入 `implementations/` 下指定的 Python class。新增策略時建立一份 YAML 規格及一個實作檔，不必在問答、評測或實驗頁分別登記。

目錄結構：

~~~text
strategies/
├── implementations/
│   └── my_strategy/strategy.py
└── specifications/
    └── my_strategy.yaml
~~~

YAML 至少要宣告穩定 `id`、UI 顯示用 `name`、實作相對路徑與 class 名稱；`parameters` 可描述型別（`string`、`integer`、`number`、`boolean`）、預設值、範圍、選項與 UI 控制型態。`implementation.file` 必須留在 `implementations/` 子目錄內。策略實作 class 要提供與既有策略相同的 `strategy_id` 和 `retrieve(context, config)` 介面。範例可參考 `strategies/specifications/vector.yaml`、`strategies/specifications/hybrid.yaml` 與對應實作。

### 建圖流程

1. 使用 PyMuPDF 讀取 PDF 文字，並排除每頁頂部與底部約 8% 的常見頁首頁尾區域。
2. 將文字切成可重疊的片段；預設片段大小為 1500 字元、重疊 200 字元，並保留頁碼。
3. 將文件分批送往建圖模型規劃 Schema。各批可同時執行，全部完成後再分層整合。
4. 依確認後的 Schema 並行抽取實體與關係，等待整批完成後統一正規化、去重及整合。
5. 分批建立原文片段、實體與關係的向量，寫入 Neo4j，並建立向量索引。

### 問答流程

送出問題時只會對「問題」建立 Embedding，不會重新計算整個圖譜的向量。系統會：

1. 只對問題建立 Embedding，使用它查詢 Neo4j 向量索引。
2. 「基本向量檢索」由 VectorCypherRetriever 取得向量相似的證據；「混合檢索」另外以問題文字查詢 CJK 全文索引，再由 HybridCypherRetriever 的 naive ranker 融合排序。
3. 若勾選「使用 Reranker」，本機 Reranker 依問題詞彙匹配、完整關鍵詞、檢索分數與證據類型重排，再選出 Top K；兩種模式會先取最多 3 倍候選（上限 50 筆），候選內容不會因此額外傳送至外部服務。
4. 選擇「混合檢索」時，可用「擴展圖譜證據」設定控制是否從命中證據擴展相關實體與關係；啟用時先回查初始來源原文，再對新擴展來源執行一次受限原文回查，第二輪不再遞迴擴展。
5. 將問題、圖譜內容及原文證據交給回答模型生成答案。

「基本向量檢索」只搜尋向量索引；「混合檢索」使用官方 HybridCypherRetriever 同時搜尋向量與全文索引，並可選擇是否使用本機 Reranker 或圖譜證據擴展。匯入階段預先建立 Embedding、向量索引與全文索引，是後續問答能快速檢索的關鍵。

自動問答測試會顯示 Recall@5、Recall@10 與 MRR：判定依據是同一 PDF 文件的來源頁碼是否命中，不要求 chunk 編號完全相同；檢索結果先合併重複的 PDF－頁碼，再以不同頁面的排名計算 Recall@5／Recall@10（前 5／10 個頁面是否命中）及 MRR（第一個命中頁面的排名倒數）。逐題結果與指標會保存至專案並在重新載入時還原。既有測試結果不會自動重跑；要以頁碼規則更新舊結果，請重新執行測試。

自動生題時，每次嘗試先從所屬 PDF 隨機抽一個焦點頁，將該頁涵蓋的原文 chunk 交給 LLM 判斷是否足以支持一個明確、可獨立回答的問答。若內容有截斷、指代不明，或條件／步驟／結論可能延續到其他頁，系統便把前後相鄰頁的 chunk 加入上下文並再次檢查；直到 LLM 判定內容完整，或已擴展至整份 PDF。跨頁 chunk 會一併納入，題目來源仍記錄實際引用的頁碼與 chunk。

向量索引會依目前 Embedding 維度命名，例如 `graph_evidence_embedding_1536`。由於 Neo4j 不允許相同標籤與屬性同時存在不同維度的向量索引，每次匯入會先刪除本工具的舊向量索引，再建立目前維度的唯一索引；不會刪除其他應用程式的索引。更換 Embedding 模型後需重新執行「Embedding 並匯入 Neo4j」。

從舊版升級時，請在介面重新執行一次「Embedding 並匯入 Neo4j」，以建立 `graph_evidence_fulltext` 全文索引及新的維度專屬向量索引；既有資料不會只因更新程式碼而自動建立索引。

## 基本使用流程

1. 在「1-0 專案設定」建立新車型專案，或載入既有專案；完成前仍可使用「1-1 連線設定」，後續工作頁會保持鎖定。可從此頁匯出專案 ZIP，或匯入既有 ZIP 封裝；匯入後會自動載入新專案。刪除專案時瀏覽器會再次要求確認，確認後才永久刪除該專案的本機設定、文件與紀錄。
2. 在「連線設定」填入 Neo4j 與 OpenAI 相容模型服務；「一鍵測試」會依序測試 Neo4j、LLM 與 Embedding 服務。測試 Neo4j 連線時會建立並驗證目前專案專屬 database。若已有專案，專案清單會預設選取一個可用專案。
3. 在 PDF 頁上傳文件並確認解析結果。
4. 前往「3. 建圖」，選擇全文或隨機 N 頁；「分析文件並規劃 Schema（可選）」可略過，留白 Schema 時會直接抽取實體與關係。
5. 視需要修改 Schema，設定 LLM 與最大並行數，再執行「確認 Schema 並抽取知識圖譜」。
6. 選擇資料處理方式，按下「Embedding 並匯入 Neo4j」。
7. 「4. 問答測試」可進行單題提問。
8. 前往「5. 自動問答測試」，指定每份 PDF 的題數及是否允許跨 PDF 並行後建立測試集。題目與答案可直接修改並自動保存，也可匯入 JSON／CSV 或匯出 JSON；每題分別保存 `question_sources` 與 `answer_sources`，兩者都是 `{document_id, document_name, pages}` 清單，因此一題可跨多份文件、各文件也可有多個頁碼。`document_id` 由 PDF 內容 SHA-256 產生；舊版頁碼／來源文件欄位仍可匯入。回答模型與評測模型可分開設定，評測模型用於判斷生成答案是否符合標準答案。可設定是否使用 Reranker 及是否擴展圖譜證據。題目在所屬車型專案的完整圖譜中檢索，不進行 PDF routing。
9. 「0-2 管理題目集」匯入中央題庫；「0-3 綁定題目集」將一份或多份題目集綁定到專案，保存後會寫入專案設定，離開頁面再返回仍會還原。舊專案題目集會在進入中央題庫／綁定頁或執行測試時遷移，原專案 JSON 保留舊資料備份。前往「1-7 單一專案實驗」選用目前專案已綁定的題目集；此頁不提供臨時匯入。跨專案實驗會讀取每個成員專案在 0-3 的綁定，並分題目集呈現結果。其餘實驗組、回答與評測設定及匯出行為照常保存。格式範例見 [`docs/題目集匯入與實驗結果格式範例.md`](docs/題目集匯入與實驗結果格式範例.md)。
10. 所有設定、參數及處理結果會在目前專案中自動保存；每次進入「1-0 專案設定」也會自動更新專案清單。

專案資料位於 `data/projects/<project-id>/`。`project.json` 保存專案設定、抽取出的實體與關係、Neo4j 匯入狀態、自動測試結果、問答紀錄與中央題目集 ID 綁定；舊版內嵌題目集會保留作遷移備份。中央題目集本體位於 `data/question_sets/<question-set-id>.json`。`documents/` 保存 PDF 副本，`exports/` 保存匯出結果；`activity.log` 為 JSON Lines，每筆以本地時區時間記錄使用者、操作名稱及結果。模型 API endpoint／API Key 僅由「0-4 API Key 設定」寫入 `.env` 並供各頁共用。專案封裝會包含專案內資料及該專案綁定的中央題目集副本，但不包含 `.env`、服務模型憑證或外部 Neo4j Database 本體。專案設定可能包含明文 Neo4j 密碼，因此 ZIP 未加密，請妥善保管、勿公開分享。

問答頁不要求在同一工作階段先建圖；只要目前專案專屬的 Neo4j database 中已有圖譜即可使用。刪除專案不會自動刪除 Neo4j database；若升級前曾使用共用 database，需將保存的抽取結果重新匯入新專案 database。

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
| `.env` | `/app/.env` | Neo4j 與 OpenAI API endpoint、帳號及 API key |
| `config/model_settings.yaml` | `/app/config/model_settings.yaml` | OpenAI 模型選擇與允許清單 |
| `data/` | `/app/data/` | 專案 JSON、PDF 副本、測試題目與問答紀錄 |

若 Neo4j 執行在 Docker 主機上，請在 `.env` 使用 `host.docker.internal`，不能使用容器自己的 `localhost`：

~~~dotenv
NEO4J_URI=bolt://host.docker.internal:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

~~~

保留 `.env.example` 中 OpenAI API Base URL，並填入自己的 API key。不要將 `.env` 提交至 Git。

### 4. 使用 Dockerfile 建置映像

~~~bash
docker build --pull -t pdf-graphrag:latest .
~~~

此命令會依 Dockerfile 執行以下工作：

1. 下載 `python:3.12-slim-bookworm` 基底映像。
2. 安裝 `requirements.txt` 中的 Python 套件。
3. 複製 `src/` 與預設 `config/`。
4. 建立容器內的 `/app/data`，開放 Gradio 的 8080 port，並設定 HTTP health check。

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
  -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/.env:/app/.env" \
  -v "$(pwd)/config/model_settings.yaml:/app/config/model_settings.yaml" \
  -v "$(pwd)/data:/app/data" \
  pdf-graphrag:latest
~~~

Windows PowerShell：

~~~powershell
docker run -d --name pdf-graphrag --restart unless-stopped -p 8080:8080 --add-host=host.docker.internal:host-gateway -v "$PWD/.env:/app/.env" -v "$PWD/config/model_settings.yaml:/app/config/model_settings.yaml" -v "$PWD/data:/app/data" pdf-graphrag:latest
~~~

參數用途：

- `-p 8080:8080`：將主機的 8080 port 對應到 Gradio。
- `--restart unless-stopped`：Docker 服務重新啟動後自動恢復容器。
- `--add-host=host.docker.internal:host-gateway`：讓 Linux 容器能連到主機上的 Neo4j。
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

狀態成為 `healthy` 後，開啟 <http://localhost:8080>。若主機 8080 已被占用，可把啟動參數改成 `-p 8081:8080`，再開啟 <http://localhost:8081>。

進入介面後，先在「0-0 使用者」選擇使用者；「0-2」管理中央題庫、「0-3」管理專案綁定、「0-4」管理共用 OpenAI API 憑證。「1-1 連線設定」及「2-1 成員專案連線測試」用於 Neo4j 診斷。API 設定寫入已掛載的 `.env`，模型選擇保存在 `config/model_settings.yaml`；未先測試連線仍可操作，請求執行時才會驗證連線。

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

若 8080 已被占用，將啟動參數改為 -p 8081:8080，再開啟 http://localhost:8081。

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

啟動腳本會記錄 `requirements.txt` 的雜湊；相依套件未變更時略過 pip 安裝，只有首次啟動或需求檔變更時才安裝。若需修復或強制重裝套件，刪除 `.venv/.requirements.sha256` 後再執行腳本。

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
python -m pip install -r requirements-dev.txt
pytest
~~~

## 限制與注意事項

- PDF 必須包含可選取的文字層；目前不提供 OCR。
- 加密 PDF 不支援。
- 建議先以少量頁面驗證 Schema、模型輸出及 Neo4j 寫入結果，再處理大型文件。
- 每次執行「Embedding 並匯入 Neo4j」都會先刪除目前 Database 中由本工具建立的圖譜；不會刪除其他標籤的資料。
- .env 可能包含 API Key 與 Neo4j 密碼，請勿提交至 Git；專案已透過 .gitignore 排除。
- PyMuPDF 採 AGPL／商業雙授權，封裝、散布或商用前請確認授權需求。
