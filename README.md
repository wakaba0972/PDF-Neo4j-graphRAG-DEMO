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
- 每個抽取請求只處理一個 chunk，以多請求並行抽取知識圖譜；來源 chunk、頁碼及文件由程式依據該請求回填，整批完成後再統一去重整合。
- 將文件片段、實體與關係建立 Embedding，寫入 Neo4j 並建立向量與全文索引。
- 每次匯入前自動清空本工具建立的既有圖譜，再寫入本次抽取結果。
- 從每份 PDF 自動建立指定數量的問題、標準答案、來源頁碼與來源 chunk；每題先隨機抽一頁，並由 LLM 檢查內容是否足以形成完整問答，不足時逐步擴展前後頁面；可選擇跨 PDF 並行生題，並支援手動編輯、自動保存、JSON／CSV 匯入與 JSON 匯出，再一鍵執行 RAG 回答及模型判分。
- 提供五種可比較的檢索策略：「基本向量檢索」、「混合檢索」、「Agentic Search」、「Agentic Search 2」與「Agentic Search 3」。Agentic Search 由 LLM 從受限圖譜鄰居中選路；Search 2 由 LLM 規劃受限唯讀 Cypher；Search 3 則由 LLM 透過工具呼叫自主選擇全文搜尋、圖譜擴展或來源 chunk 查詢，直到停止或達到呼叫上限。Search 3 工具由程式固定實作，不接受任意 Cypher。策略可設定各自參數，並搭配共用 Reranker 與證據擴展。
- 2-2 跨專案實驗的 AI 評測完成後可保存不可變執行快照；3-0 依實驗組／成員專案／題目集分層抽樣並平均分派給五位使用者，3-1 收集盲評；4-0 對照同一批題目的 AI 與人工評分。
- 單一專案與跨專案實驗均可設定實驗組、回答模型、檢索策略與並行數；摘要會按實驗組統計答題成績及三種檢索來源指標。人工評測抽樣支援固定題數與按題數比例分配。
- 模型與 Embedding 僅支援 OpenAI API，相容的 OpenAI API Base URL 亦可設定。連線測試只用於診斷，不是後續操作的前置條件；實際呼叫失敗時會回報 API 錯誤。
- 「0-0 使用者」先選擇目前操作人（57、58、59、60、61）；選擇後才能進入其他頁面。「0-1 管理題目集」只匯入及管理中央題庫；「0-2 綁定題目集」設定各專案使用哪些題目集；「0-3 API Key 設定」提供共用 OpenAI API Base URL／Key，供對話、建圖及 Embedding 使用。使用者下拉選擇供操作標記與稽核使用，並非帳號登入或身分驗證。

Neo4j 專案隔離需使用支援多資料庫的 Neo4j Enterprise，並提供可在 `system` database 建立 database 的管理權限；Neo4j Community 與 Aura 不支援此種每專案獨立 database 建立方式。

## 實作原理

### 頁面功能導覽

| 頁面 | 用途 |
|---|---|
| 0-0 使用者 | 選擇操作標記使用者；這不是登入或權限驗證。 |
| 0-1 管理題目集 | 匯入、預覽、刪除中央題庫中的題目集。刪除前會提示目前綁定的專案。 |
| 0-2 綁定題目集 | 管理各單一專案使用哪些中央題目集。 |
| 0-3 API Key 設定 | 設定所有頁面共用的 OpenAI API URL 與 Key。 |
| 1-0～1-6 單一專案流程 | 管理車型專案、連線、PDF、建圖、問答、自動問答與單一專案實驗。 |
| 2-0～2-2 跨專案實驗 | 建立實驗專案、加入已建圖成員專案，並以各成員專案綁定的題目集執行同一批實驗組。 |
| 3-0～3-1 人工評測 | 對已保存的 AI 評測執行分層抽樣、分派並收集人工評分。 |
| 4-0 AI／人工評測結果 | 查看 AI 與人工評分配對結果、一致性及檢索來源指標。 |

各單一專案頁面依序為：1-0 專案設定、1-1 Neo4j 連線設定、1-2 PDF 與建圖參數、1-3 建圖、1-4 問答測試、1-5 自動問答測試、1-6 單一專案實驗。跨專案流程為 2-0 實驗專案、2-1 成員專案連線測試、2-2 自動實驗測試。

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
3. 直接依文件文本抽取實體與關係；Schema 規劃已移除。使用者可選擇提供自訂 Schema 約束，也可留白讓模型自行從文本辨識類型。
4. 以「一個 chunk 一個請求」並行抽取；來源 chunk、頁碼及文件由程式記錄，不依賴模型回報。等待整批完成後統一正規化、去重及整合。
5. 分批建立原文片段、實體與關係的向量，寫入 Neo4j，並建立向量索引。

### 問答流程

送出問題時只會對「問題」建立 Embedding，不會重新計算整個圖譜的向量。系統會：

1. 只對問題建立 Embedding，使用它查詢 Neo4j 向量索引。
2. 「基本向量檢索」以 VectorCypherRetriever 取回向量相似證據；「混合檢索」另查 CJK 全文索引，再由 HybridCypherRetriever 融合排序。
3. 「Agentic Search」以向量命中 chunk 對應的實體為起點。程式查詢真實 `EXTRACTED_RELATION` 鄰居，LLM 從受限候選選擇要擴展的路徑；LLM 不生成 Cypher。
4. 「Agentic Search 2」同樣先取向量起始證據，再由 LLM 多輪規劃唯讀 Cypher；程式驗證並限制查詢範圍，執行後把新證據與搜尋歷程交回模型，直到模型停止或輪數／時間上限到達。
5. 「Agentic Search 3」先取向量起始證據，再讓 LLM 透過工具呼叫選擇全文搜尋、圖譜擴展或取回來源 chunk；工具是固定的唯讀程式，不會執行模型生成的 Cypher。搜尋後由既有回答流程根據證據生成答案。
6. 可選「Reranker」使用本機字詞與檢索分數排序；「LLM Reranker」以已設定的模型判斷候選相關性。證據擴展是另一個獨立選項，用實體／關係連結補回原文。所有策略與後處理最後都受 Top K 控制。
7. 將問題、取回的圖譜資訊及原文證據交給回答模型生成答案。Luna 的模型推理強度是單次模型呼叫設定；它與 Agentic 策略的多輪搜尋流程是不同設定。

1-3 的「圖譜範圍」可選擇將所有 PDF 合併成單一圖譜，或按 `document_id` 為每份 PDF 建立獨立的邏輯圖譜。後者仍在同一個專案 Neo4j database 中保存多個 `GraphDocument`；實體、關係、證據與圖邊都限定在所屬文件，不會因名稱或 chunk 編號相同而跨文件合併。Vector、Hybrid 與 Agentic 策略仍預設搜尋專案全部文件的證據。文件分區模式會逐份 PDF 抽取，因此可能增加整體抽取時間；切換圖譜範圍後必須重新抽取並匯入。所有文件仍共用專案的 Embedding 模型與向量維度。此分區可作為未來「每份文件一個搜尋 Agent」的安全工作範圍，但目前尚未加入多 Agent 派工與答案整合。

檢索成效以題目集人工標註的答案來源文件與頁碼計算。Evidence Hit Rate 是有 Gold 的題目中，至少命中一個 Gold 頁面的比例；Evidence Precision 是先逐題計算「命中的不同文件－頁碼數／取回的不同文件－頁碼數」，再對有 Gold 頁碼的題目取算術平均；Document Hit Rate 是至少取回一份 Gold 文件的題目比例。文件身分以文件 ID 優先，沒有 ID 時才用文件名稱；chunk ID 不參與比對。同一文件同一頁由多個 chunk 取回只計一次。有 Gold 頁碼但沒有取回來源時 Precision 為 0；沒有 Gold 頁碼的題目不列入檢索指標平均。

自動生題時，每次嘗試先從所屬 PDF 隨機抽一個焦點頁，將該頁涵蓋的原文 chunk 交給 LLM 判斷是否足以支持一個明確、可獨立回答的問答。若內容有截斷、指代不明，或條件／步驟／結論可能延續到其他頁，系統便把前後相鄰頁的 chunk 加入上下文並再次檢查；直到 LLM 判定內容完整，或已擴展至整份 PDF。跨頁 chunk 會一併納入，題目來源仍記錄實際引用的頁碼與 chunk。

向量索引會依目前 Embedding 維度命名，例如 `graph_evidence_embedding_1536`。由於 Neo4j 不允許相同標籤與屬性同時存在不同維度的向量索引，每次匯入會先刪除本工具的舊向量索引，再建立目前維度的唯一索引；不會刪除其他應用程式的索引。更換 Embedding 模型後需重新執行「Embedding 並匯入 Neo4j」。

從舊版升級時，請在介面重新執行一次「Embedding 並匯入 Neo4j」，以建立 `graph_evidence_fulltext` 全文索引及新的維度專屬向量索引；既有資料不會只因更新程式碼而自動建立索引。

## 基本使用流程

1. 在「0-0 使用者」選擇操作標記，再到「0-3 API Key 設定」設定共用 OpenAI API URL／Key。此使用者清單不是登入或權限控管。
2. 在「1-0 專案設定」建立或載入車型專案，也可在此匯入／匯出專案 ZIP。專案使用自己的 Neo4j database；建立多資料庫需 Neo4j Enterprise 與管理權限。
3. 在「1-1 連線設定」設定及測試 Neo4j，然後到「1-2 PDF 與參數」上傳 PDF、確認 chunking 與建圖參數。
4. 到「1-3 建圖」選擇合併所有文件或依 `document_id` 分區，再直接從文本抽取實體與關係；若不需要自訂 Schema，保持欄位空白即可。檢查結果後按「Embedding 並匯入 Neo4j」。Embedding 請求和圖資料寫入可分批並設定最大並行數。
5. 「1-4 問答測試」用單題檢查檢索與回答；「1-5 自動問答測試」可由 PDF 生題，或載入已準備的題目集後分開執行回答與評測。人工評分採 0（錯）、1（部分正確）、2（全對）。
6. 在「0-1 管理題目集」匯入、預覽或刪除中央題目集；每筆具名來源需包含非空正整數 `pages`。沒有 Gold 的題目以空 `answer_sources` 表示。於「0-2 綁定題目集」指定每個專案使用哪些題目集，之後「1-6 單一專案實驗」及跨專案測試會使用這些綁定。此頁支援建立多個實驗組，直接在組列設定模型、檢索策略、策略專屬參數、Top K、Reranker 與證據擴展。
7. 建立「2-0 實驗專案」並加入多個已建圖專案，在「2-1」診斷成員專案 Neo4j 連線，再到「2-2 自動實驗測試」執行跨專案實驗。每組套用到所有成員專案，題目仍按各專案／題目集分開統計；完成 AI 評測後按「保存實驗結果」建立不可變快照。
8. 「3-0 人工評測抽樣」可按每個分層固定題數，或依總題數比例抽樣，並平均分派給使用者；「3-1 人工評測」收集盲評。「4-0 AI／人工評測結果」比較配對分數、一致性及 Evidence Hit Rate、Evidence Precision、Document Hit Rate。格式見 [`docs/題目集匯入與實驗結果格式範例.md`](docs/題目集匯入與實驗結果格式範例.md)。

專案資料位於 `data/projects/<project-id>/`。大型 chunks 與知識圖譜抽取資料分別保存於壓縮 sidecar 檔，避免每次載入專案都讀取整份資料；`project.json` 保存輕量設定、建圖狀態、測試／實驗資料與題目集綁定。中央題目集位於 `data/question_sets/`。跨專案實驗位於 `data/experiment_projects/`，已保存的 AI 執行快照會供人工抽樣及 AI／人工比較使用。`activity.log` 記錄操作使用者、時間與事件。API endpoint／Key 由「0-3 API Key 設定」寫入 `.env`，所有頁面共用。專案 ZIP 不包含 `.env` 或外部 Neo4j Database；請注意封裝內的專案設定可能含 Neo4j 密碼。

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

進入介面後，先在「0-0 使用者」選擇使用者；「0-1」管理中央題庫、「0-2」管理專案綁定、「0-3」管理共用 OpenAI API 憑證。「1-1 連線設定」及「2-1 成員專案連線測試」用於 Neo4j 診斷。API 設定寫入已掛載的 `.env`，模型選擇保存在 `config/model_settings.yaml`；未先測試連線仍可操作，請求執行時才會驗證連線。

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
