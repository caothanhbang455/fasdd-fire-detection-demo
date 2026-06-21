# Deployment Guide — FASDD-CV Demo

Hai repo cần deploy:

| Repo | Mục đích | URL sau deploy |
|---|---|---|
| `github_pages/` | Academic project page (static) | `https://USERNAME.github.io/fasdd-cv-page` |
| `hf_spaces/`    | Live demo (Gradio + GPU)       | `https://USERNAME-fasdd-cv-demo.hf.space` |

---

## Phần 1 — HuggingFace Spaces (Demo)

### Bước 1: Tạo Space

1. Vào https://huggingface.co/new-space
2. Điền:
   - **Space name**: `fasdd-cv-demo`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: `T4 medium` (free tier, 8h/ngày GPU)
3. Click **Create Space**

### Bước 2: Upload source code

```bash
# Clone repo Space vừa tạo
git clone https://huggingface.co/spaces/YOUR_USERNAME/fasdd-cv-demo
cd fasdd-cv-demo

# Copy toàn bộ hf_spaces/ vào đây
cp -r /path/to/hf_spaces/* .

# Commit + push
git add .
git commit -m "Initial deploy: YOLO11m-seg + FLAME pipeline"
git push
```

Hoặc dùng HF web UI: **Files tab** → **Add file** → Upload từng file.

### Bước 3: Upload model checkpoints

Model `.pt` files thường > 100MB, không commit vào git. Upload riêng:

**Cách A — HF web UI (dễ nhất):**
1. Vào Space của bạn → **Files** tab
2. Click **Add file** → **Upload files**
3. Upload vào thư mục `models/`:
   - `models/det_best.pt`
   - `models/seg_v1_best.pt`
   - `models/seg_v2_best.pt`

**Cách B — huggingface_hub Python:**
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="path/to/det_best.pt",
    path_in_repo="models/det_best.pt",
    repo_id="YOUR_USERNAME/fasdd-cv-demo",
    repo_type="space",
)
```

**Cách C — git-lfs (nếu muốn track bằng git):**
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add models/det_best.pt models/seg_v1_best.pt models/seg_v2_best.pt
git commit -m "Add model checkpoints"
git push
```

### Bước 4: Verify deploy

Space sẽ build tự động (mất ~2-3 phút). Khi thấy **Running** (xanh):
1. Mở URL `https://YOUR_USERNAME-fasdd-cv-demo.hf.space`
2. Upload một video test
3. Kiểm tra dropdown có 3 checkpoint không
4. Chạy thử với FLAME enabled

**Nếu bị lỗi:**
- Xem **Logs** tab trong Space để debug
- Lỗi thường gặp: thiếu `.pt` file → kiểm tra path trong `models/`
- `supervision` version conflict → pin `supervision==0.21.0` trong requirements.txt

### Bước 5 (optional): Thêm example videos

Đặt video test vào `examples/` folder, sau đó sửa `app.py`:

```python
gr.Examples(
    examples=[
        ["examples/fire_test.mp4"],
        ["examples/smoke_test.mp4"],
        ["examples/neither_sunset.mp4"],   # FP case — để show FLAME hoạt động
    ],
    inputs=[video_input],
    label="Example Surveillance Clips",
)
```

---

## Phần 2 — GitHub Pages (Project Page)

### Bước 1: Tạo repo

1. Vào https://github.com/new
2. **Repository name**: `fasdd-cv-page`
3. **Public** (bắt buộc cho Pages free)
4. **Add a README file**: check
5. Click **Create repository**

### Bước 2: Upload index.html

**Cách A — web UI:**
1. Vào repo → **Add file** → **Upload files**
2. Upload `github_pages/index.html`
3. Commit message: "Add project page"
4. Click **Commit changes**

**Cách B — git:**
```bash
git clone https://github.com/YOUR_USERNAME/fasdd-cv-page
cd fasdd-cv-page
cp /path/to/github_pages/index.html .
git add index.html
git commit -m "Add project page"
git push
```

### Bước 3: Enable GitHub Pages

1. Repo → **Settings** → **Pages** (sidebar trái)
2. **Source**: `Deploy from a branch`
3. **Branch**: `main` → `/ (root)`
4. Click **Save**
5. Đợi ~2 phút → URL xuất hiện: `https://YOUR_USERNAME.github.io/fasdd-cv-page`

### Bước 4: Điền link thật vào index.html

Mở `index.html`, tìm và thay 4 chỗ `href="#"`:

```html
<!-- Tìm -->
<a href="#" class="badge badge-primary">   ← Report
<a href="#" class="badge">                 ← GitHub (code repo)
<a href="#" class="badge">                 ← Demo
<a href="#" class="badge">                 ← HF Space

<!-- Thay bằng -->
<a href="https://drive.google.com/YOUR_REPORT.pdf" ...>   ← Report PDF
<a href="https://github.com/YOUR_USERNAME/fasdd-cv" ...>  ← Code repo
<a href="https://YOUR_USERNAME-fasdd-cv-demo.hf.space" ...> ← Demo
<a href="https://huggingface.co/spaces/YOUR_USERNAME/fasdd-cv-demo" ...> ← HF Space
```

Cũng update demo section (cuối page):
```html
<a href="https://YOUR_USERNAME-fasdd-cv-demo.hf.space" ...>
```

Commit lại → Pages tự động rebuild trong 30 giây.

---

## Tổng kết URL sau deploy

```
📄 Project Page : https://YOUR_USERNAME.github.io/fasdd-cv-page
🤗 Demo         : https://YOUR_USERNAME-fasdd-cv-demo.hf.space
💻 Code repo    : https://github.com/YOUR_USERNAME/fasdd-cv
```

---

## Checklist

### HuggingFace Spaces
- [ ] Space created với SDK = Gradio, Hardware = T4
- [ ] `app.py`, `requirements.txt`, `README.md` uploaded
- [ ] `flame/` folder uploaded (4 files)
- [ ] `models/det_best.pt` uploaded
- [ ] `models/seg_v1_best.pt` uploaded  
- [ ] `models/seg_v2_best.pt` uploaded
- [ ] Space status = **Running** (xanh)
- [ ] Demo test với video upload thành công
- [ ] FLAME toggle hoạt động đúng

### GitHub Pages
- [ ] Repo created (Public)
- [ ] `index.html` uploaded
- [ ] Pages enabled (Settings → Pages)
- [ ] URL accessible: `https://USERNAME.github.io/fasdd-cv-page`
- [ ] 4 badge links đã điền link thật
- [ ] Demo link trong page trỏ đúng vào HF Space

---

## Troubleshooting thường gặp

**Space không start / lỗi import:**
```
ModuleNotFoundError: No module named 'supervision'
```
→ Kiểm tra `requirements.txt` có `supervision>=0.21.0`

**Model load fail:**
```
Model not found: models/seg_v2_best.pt
```
→ Kiểm tra file có trong `models/` folder trên HF Files tab

**Video processing quá chậm:**
→ Đảm bảo Space dùng T4 GPU. CPU-only Space sẽ mất 10-20x lâu hơn.

**FLAME không suppress FP:**
→ Tăng `bg_warmup_frames` lên 100 nếu video dài. Giảm `min_fg_ratio` trong `flame/background.py` nếu camera có rung nhẹ.

**GitHub Pages hiển thị 404:**
→ Đảm bảo file tên chính xác là `index.html` (không phải `Index.html`)
→ Đảm bảo Pages branch = `main` và folder = `/ (root)`
