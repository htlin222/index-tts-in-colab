# index-tts-in-colab

開一個 GitHub Issue → GitHub Action 解析內容 → 呼叫 [Colab CLI](https://github.com/googlecolab/google-colab-cli) 在免費 T4 GPU 上跑 [IndexTTS-2](https://github.com/index-tts/index-tts) zero-shot 聲音克隆 → 把合成好的 `.wav` 發成 GitHub Release，並在 issue 留言連結。

## 使用方式

1. 開一個新 issue，選 **🎙️ 語音合成請求** 範本
2. 填要朗讀的文字（每行一段）、情緒、情緒強度
3. 送出後等 Action 跑完（約 15–25 分鐘，第一次含環境建置與模型下載）
4. Issue 底下會自動留言 Release 連結，issue 自動關閉

也可以用 **Actions → synthesize → Run workflow** 手動觸發，不用開 issue。

## 架構

```
issue (opened, label=synthesize)
  │
  ▼
scripts/parse_issue.py      ─→ batch.jsonl（每行一段，情緒向量套用同一預設）
  │
  ▼
colab CLI（GitHub Secrets 裡的 ADC 憑證）
  │  colab new -s ci-<run_id> --gpu T4
  │  colab upload  ref.wav / batch.jsonl
  │  colab exec    colab_job/synthesize.py   ← 在 Colab VM 上跑 indextts2 batch
  │  colab download output.wav
  │  colab stop
  ▼
gh release create + gh issue comment/close
```

## 已知限制 / 之後可以做的事

- **同時只能有一個 GPU runtime**（Colab 免費帳號限制）。Workflow 用 `concurrency` group 序列化多個 issue，但如果有人開著瀏覽器裡的互動式 Colab notebook 占用 T4，CI 會直接失敗（`TooManyAssignmentsError`）。發生時去 [colab.research.google.com](https://colab.research.google.com) → 執行階段 → 管理工作階段 把它斷開。
- **每次 run 都是全新 VM**，環境建置＋模型下載（~6GB）沒有快取，每次都要重來一遍，佔掉大部分時間。要加速可以考慮把 venv/checkpoints 打包存 Drive，run 開始時解壓——目前先不做，避免過早優化。
- **模型下載速度不穩定**：2026-08-16 實測過一次匿名下載卡了 28 分鐘沒完成（HuggingFace 對未登入請求的限速本來就不保證）。設定 `HF_TOKEN` secret（見下）能大幅改善這個問題。`colab_job/synthesize.py` 內建每一步自己的 timeout＋定期心跳輸出，卡住時能立刻看出是哪一步、卡了多久，而不是像修這個問題之前那樣整段 30 分鐘無聲無息。
- **參考聲音固定**：用的是 repo 裡 `assets/ref_voice.wav`（7 秒乾淨人聲，已做響度正規化）。目前 v1 不支援每個 issue 換一個參考音檔，如果要換，直接替換這個檔案再 commit。
- **情緒是整段套用同一個預設**，不支援每行不同情緒。這 6 個預設向量寫死在 `scripts/parse_issue.py` 的 `EMOTION_PRESETS`。
- **一次最多 40 行 / 800 字**，超過會直接失敗（不做靜默截斷）。這個字數上限是照實測吞吐率反推的，不是隨便選的：從兩次真實 run 回歸出 batch 合成時間 ≈ 131s 固定成本（模型載入）+ 1.18s/字。舊上限是 4000 字，換算下來光合成就要 ~80 分鐘，遠遠超過 pipeline 的所有 timeout；改成 800 字後，正常情況（~25 分鐘）跟下載卡住需要重試的情況（~42 分鐘）都能在現有 timeout 內跑完。`colab_job/synthesize.py` 的 `batch_timeout` 現在也改成看總字數算，而不是看行數——行數不能反映真正的合成成本，幾行超長文字會被大幅低估時間。

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
