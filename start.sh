#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/4] 檢查 Python 虛擬環境..."
if [[ ! -x ".venv/bin/python" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[錯誤] 找不到 python3，請先安裝 Python 3.11 以上版本。" >&2
        exit 1
    fi
    python3 -m venv .venv
fi

echo "[2/4] 檢查並安裝必要套件..."
REQUIREMENTS_HASH="$(".venv/bin/python" -c \
    'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    requirements.txt)"
REQUIREMENTS_MARKER=".venv/.requirements.sha256"
if [[ ! -f "$REQUIREMENTS_MARKER" ]] || [[ "$(<"$REQUIREMENTS_MARKER")" != "$REQUIREMENTS_HASH" ]]; then
    if ! ".venv/bin/python" -m pip install -r requirements.txt; then
        echo "[錯誤] 套件安裝失敗，請檢查網路連線與上方錯誤訊息。" >&2
        exit 1
    fi
    printf '%s\n' "$REQUIREMENTS_HASH" > "$REQUIREMENTS_MARKER"
else
    echo "套件需求未變更，略過安裝。若需強制重裝，刪除 $REQUIREMENTS_MARKER 後重跑。"
fi

echo "[3/4] 檢查環境設定..."
if [[ ! -f ".env" ]]; then
    cp ".env.example" ".env"
fi

echo "[4/4] 啟動網站..."
echo "服務監聽：0.0.0.0:8080"
echo "本機網址：http://127.0.0.1:8080"
echo "按 Ctrl+C 可停止服務。"
exec ".venv/bin/python" src/app.py
