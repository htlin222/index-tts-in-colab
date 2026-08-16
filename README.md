# index-tts-in-colab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![synthesize](https://github.com/htlin222/index-tts-in-colab/actions/workflows/synthesize.yml/badge.svg)](https://github.com/htlin222/index-tts-in-colab/actions/workflows/synthesize.yml)
[![IndexTTS-2](https://img.shields.io/badge/model-IndexTTS--2.5-blue)](https://github.com/index-tts/index-tts)
[![Colab CLI](https://img.shields.io/badge/runtime-Colab%20CLI-orange)](https://github.com/googlecolab/google-colab-cli)

**Open a GitHub issue, get zero-shot voice-cloned speech back as a Release.** This repo wires up a full serverless text-to-speech pipeline entirely on free infrastructure: a GitHub Issue form triggers a GitHub Actions workflow, which provisions a free Google Colab T4 GPU via the official [Colab CLI](https://github.com/googlecolab/google-colab-cli), runs [IndexTTS-2.5](https://github.com/index-tts/index-tts) zero-shot voice cloning, and publishes the resulting `.wav` as a GitHub Release — no server, no persistent infra, no manual steps. Long text is automatically split into chunks and processed on the same warm Colab session so the ~10-minute environment/model-download cost is paid once per request, not once per chunk. See the [architecture](#架構) and [known limitations](#已知限制--之後可以做的事) sections below (Traditional Chinese) for the full design.

開一個 GitHub Issue → GitHub Action 解析內容 → 呼叫 [Colab CLI](https://github.com/googlecolab/google-colab-cli) 在免費 T4 GPU 上跑 [IndexTTS-2.5](https://github.com/index-tts/index-tts) zero-shot 聲音克隆 → 把合成好的 `.wav` 發成 GitHub Release，並在 issue 留言連結。

## 目錄

- [使用方式](#使用方式)
- [架構](#架構)
- [已知限制 / 之後可以做的事](#已知限制--之後可以做的事)
- [一次性設定（repo owner 才需要做）](#一次性設定repo-owner-才需要做)

## 使用方式

1. 開一個新 issue，選 **🎙️ 語音合成請求** 範本
2. 填要朗讀的文字（每行一段）、情緒、情緒強度
3. 送出後等 Action 跑完（環境建置＋模型下載約 15 分鐘，之後每個 chunk 再加幾分鐘，見下）
4. Issue 底下會自動留言 Release 連結，issue 自動關閉

也可以用 **Actions → synthesize → Run workflow** 手動觸發，不用開 issue。

## 架構

長文字會被切成多個 chunk，但環境建置＋模型下載只在整個 issue 處理過程中付一次，不是每個 chunk 各付一次——這是切 chunk 的重點，不然每塊都重裝一次環境會非常浪費：

```
issue (opened, label=synthesize)
  │
  ▼
scripts/parse_issue.py  ─→ batch_chunk_0.jsonl, batch_chunk_1.jsonl, ...
                             （依標點計算好每行的停頓，切塊前就算好，
                              chunk 邊界的停頓自然正確）
  │
  ▼
colab CLI（同一個 Colab session，GitHub Secrets 裡的 ADC 憑證）
  │
  │  colab new -s ci-<run_id> --gpu T4              ← 只跑一次
  │  colab upload  ref.wav / _common.py / _synth_inner_25.py / hf_token  ← 只跑一次
  │  colab exec    colab_job/setup.py               ← 只跑一次：clone + uv sync + 下模型
  │
  │  for each chunk:                                ← 重複 N 次
  │    colab upload  chunk_index.txt / batch_chunk_N.jsonl
  │    colab exec    colab_job/synth_chunk.py        （呼叫 _synth_inner_25.py，模型已在硬碟上）
  │
  │  colab exec    colab_job/concat_chunks.py        ← 只跑一次：把 chunk_0..N.wav 接起來
  │  colab download output.wav
  │  colab stop
  ▼
gh release create + gh issue comment/close
```

`colab exec -f script.py` 是把腳本內容當程式碼送進遠端 kernel 執行，不是把檔案放到 VM 硬碟上，所以 `setup.py` / `synth_chunk.py` 之間不能直接互相 import——共用的 `run()`（timeout／心跳／重試邏輯）額外用 `colab upload` 放成 `/content/_common.py`，兩支腳本都從那裡 import。

用的是 **IndexTTS-2.5**（`IndexTeam/IndexTTS-2.5`），不是 2.0。官方 `indextts2` CLI 只包了 `infer_v2.py`（2.0），沒有包 `infer_v2_5.py`——`indextts2 download`/`indextts2 batch` 都是寫死抓 2.0。所以 `setup.py` 自己呼叫 index-tts 內部的 `snapshot_download`/`ensure_models_available` 抓 2.5 權重，`synth_chunk.py` 也不走 `indextts2 batch`，改成呼叫另外上傳的 `_synth_inner_25.py`（在 uv venv 內直接 `import indextts.infer_v2_5.IndexTTS2`，逐行呼叫 `.infer()`）。順便省了 2.0 版必抓、但我們用不到的 QwenEmotion 模型（~1.2GB，只有 `emo_text` 模式才需要，我們一直只用 `emo_vector`）。

## 已知限制 / 之後可以做的事

- **同時只能有一個 GPU runtime**（Colab 免費帳號限制）。Workflow 用 `concurrency` group 序列化多個 issue，但如果有人開著瀏覽器裡的互動式 Colab notebook 占用 T4，CI 會直接失敗（`TooManyAssignmentsError`）。發生時去 [colab.research.google.com](https://colab.research.google.com) → 執行階段 → 管理工作階段 把它斷開。
- **每次 run 都是全新 VM**，環境建置＋模型下載（~6GB）沒有快取，每次都要重來一遍，佔掉大部分時間。要加速可以考慮把 venv/checkpoints 打包存 Drive，run 開始時解壓——目前先不做，避免過早優化。
- **模型下載速度不穩定**：2026-08-16 實測過一次匿名下載卡了 28 分鐘沒完成（HuggingFace 對未登入請求的限速本來就不保證）。設定 `HF_TOKEN` secret（見下）能大幅改善這個問題。`colab_job/_common.py` 的 `run()` 內建每一步自己的 timeout＋定期心跳輸出，卡住時能立刻看出是哪一步、卡了多久，而不是整段安靜無聲。逾時時用 SIGTERM（給 15 秒清理機會）而不是直接 SIGKILL——直接強殺過一次，導致 huggingface_hub 的快取鎖檔沒清乾淨，下次重試立刻卡死。
- **`google-colab-cli` 0.6.0 的 `colab exec` 不會把遠端程式碼執行失敗回報成失敗**：讀過原始碼確認，它只檢查自己本地拋出的例外（例如外層 `--timeout` 逾時），完全不檢查遠端 kernel 執行的結果——腳本裡的 `sys.exit(1)` 會被靜默吞掉，`colab exec` 照樣回傳 0。這代表 workflow 不能只看 `colab exec` 的 exit code 判斷成敗；每次呼叫都要把輸出 `tee` 出來，自己 grep `FATAL:` 字樣，找到才讓 step 失敗（見 `.github/workflows/synthesize.yml`）。這是 2026-08-16 用真實多 chunk 測試才抓到的——第一次跑的時候模型根本沒下載完，但 workflow 顯示全綠。
- **參考聲音固定**：用的是 repo 裡 `assets/ref_voice.wav`（7 秒乾淨人聲，已做響度正規化）。目前 v1 不支援每個 issue 換一個參考音檔，如果要換，直接替換這個檔案再 commit。
- **情緒是整段套用同一個預設**，不支援每行不同情緒。這 6 個預設向量寫死在 `scripts/parse_issue.py` 的 `EMOTION_PRESETS`。
- **行與行之間的拼接**：`synth_chunk.py` 不用 `indextts2 batch --concat`（那個是直接把靜音貼在滿振幅音訊旁邊，聽起來像硬切/卡頓），改成逐行各自輸出、用 `_common.py` 的 `concat_wavs_with_fade()` 自己拼接，每段頭尾各加 20ms 淡入淡出——停頓長度不變（還是照標點計算），只是把數位懸崖式的切點磨掉。
- **一次最多 150 行 / 3000 字，單行最多 200 字**，超過會直接失敗（不做靜默截斷）。這個上限是照實測吞吐率算的，不是隨便選的：從兩次真實 run 回歸出 batch 合成時間 ≈ 131s 固定成本（模型載入）+ 1.18s/字。切 chunk 之後總時間會隨字數線性增加，不再有 800 字這種硬牆——但還是需要一個上限，否則一篇超長文章會讓單一 issue 跑好幾小時，吃掉大量 Colab 免費運算額度。3000 字大約切成 4-5 個 chunk，總耗時抓 ~1.5-2 小時。
- **chunk 大小固定 700 字**（`CHUNK_MAX_CHARS`），每個 chunk 各自付一次「模型重新載入 GPU」的成本（約 131 秒），但環境建置和模型下載只在整個 issue 處理過程付一次。

## 一次性設定（repo owner 才需要做）

### `COLAB_ADC_CREDENTIALS`（必要）

Colab CLI 用你 Google 帳號的 [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) 認證。這把憑證存進 GitHub Secrets 後，CI 就能用你的身分開 Colab GPU session——這代表：

- CI 會消耗你 Google 帳號的 Colab 免費運算額度
- 這把憑證的 scope 含 `cloud-platform`，範圍不小；repo 若被別人拿到寫入權限（例如接受了惡意 PR 改 workflow），這把憑證等於被盜用
- 想收回權限，去 [Google 帳號權限頁](https://myaccount.google.com/permissions) 撤銷 `colab-cli` 這個 OAuth 應用程式，並在 GitHub 上刪掉 `COLAB_ADC_CREDENTIALS` secret

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

gh secret set COLAB_ADC_CREDENTIALS < ~/.config/gcloud/application_default_credentials.json
```

### `HF_TOKEN`（選填，但強烈建議）

匿名對 HuggingFace Hub 發請求會被限速，速度不保證（實測卡過 28 分鐘沒下完 ~6GB 的模型）。去 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 拿一個 **read** 權限的 token，存進 secret：

```bash
gh secret set HF_TOKEN --repo htlin222/index-tts-in-colab
# 貼上 token，按 Ctrl-D
```

沒設這個 secret 不會讓 pipeline 失敗，只是下載會退回匿名、變慢變不穩。

## License

這個 repo 自己的程式碼（workflow / issue 表單 / `scripts/`、`colab_job/` 底下的腳本）採用 [MIT License](LICENSE)。它在 runtime clone 的 [index-tts](https://github.com/index-tts/index-tts) 原始碼和從 [HuggingFace 下載的 IndexTTS-2.5 模型權重](https://huggingface.co/IndexTeam/IndexTTS-2.5)各自有自己的授權（Bilibili 自訂授權，不是 MIT），不在這個 MIT 範圍內。
