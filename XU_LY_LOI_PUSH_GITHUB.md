# Xử lý lỗi: File quá lớn không push được lên GitHub

## ❌ Lỗi gặp phải

```
remote: error: File features.pkl is 126.90 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage
```

**Nguyên nhân**: File `features.pkl` (126.90 MB) và các file khác vượt quá giới hạn 100 MB của GitHub.

---

## ✅ GIẢI PHÁP: Xóa .git và làm lại từ đầu

### Bước 1: Xóa thư mục .git hiện tại

```powershell
cd "d:\gitHub\Hẹ hẹ hẹ (Học sâu)"

# Xóa thư mục .git (xóa toàn bộ lịch sử Git)
Remove-Item -Recurse -Force .git
```

### Bước 2: Kiểm tra file .gitignore đã được cập nhật

File `.gitignore` đã được cập nhật để loại trừ:
- ✅ `*.pkl` - Feature files (126+ MB)
- ✅ `*.pth` - Model weights (100+ MB)
- ✅ `*.jpg, *.png` - Dataset images (GB)
- ✅ TensorBoard logs, Python cache, LaTeX temp files

```powershell
# Xem nội dung .gitignore
Get-Content .gitignore
```

### Bước 3: Khởi tạo Git repository mới

```powershell
# Tạo Git repo mới
git init

# Kiểm tra branch name (nên là 'main')
git branch
```

### Bước 4: Cấu hình Git (nếu chưa có)

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Bước 5: Thêm file vào staging (Git sẽ tự động bỏ qua files trong .gitignore)

```powershell
# Add tất cả files (trừ những files trong .gitignore)
git add .

# Kiểm tra những file sẽ được commit
git status
```

**Xác nhận**: Đảm bảo KHÔNG thấy các file sau trong `git status`:
- ❌ `features.pkl`, `features_*.pkl`
- ❌ `*.pth` (model weights)
- ❌ `*.jpg`, `*.png` (images)
- ❌ `__pycache__/`
- ❌ `runs/`, `Dec*/` (TensorBoard logs)

### Bước 6: Commit lần đầu

```powershell
git commit -m "Initial commit: Image Captioning project with EfficientNet and ResNet50"
```

### Bước 7: Kết nối với GitHub repository

```powershell
# Thêm remote (repo đã tạo trên GitHub)
git remote add origin https://github.com/kagamikuro1024/Image-Captioning-_-Deep-Learning.git

# Kiểm tra remote
git remote -v
```

### Bước 8: Đổi branch thành main

```powershell
git branch -M main
```

### Bước 9: Push lên GitHub

```powershell
# Push lần đầu
git push -u origin main
```

**Nhập thông tin đăng nhập**:
- Username: `kagamikuro1024`
- Password: **Personal Access Token** (KHÔNG phải mật khẩu!)

---

## 🔍 Kiểm tra trước khi push

### Kiểm tra kích thước repository

```powershell
# Xem tổng dung lượng Git repo (nên < 300 MB)
Get-ChildItem .git -Recurse | Measure-Object -Property Length -Sum | Select-Object @{Name="Size(MB)";Expression={$_.Sum / 1MB}}

# Liệt kê các file lớn nhất trong staging
git ls-files | ForEach-Object { 
    $size = (Get-Item $_).Length / 1MB
    if ($size -gt 5) {
        [PSCustomObject]@{
            File = $_
            "Size(MB)" = [math]::Round($size, 2)
        }
    }
} | Sort-Object "Size(MB)" -Descending
```

### Kiểm tra files đã được ignore

```powershell
# Xem các file bị ignore
git status --ignored

# Hoặc kiểm tra cụ thể
git check-ignore -v features.pkl
git check-ignore -v *.pth
```

---

## 📊 Files NÊN và KHÔNG NÊN push

### ✅ NÊN PUSH (< 50 MB tổng cộng)

- ✅ `*.py` - Source code (~500 KB)
- ✅ `*.ipynb` - Notebooks (~10 MB)
- ✅ `README.md` - Documentation (~30 KB)
- ✅ `requirements.txt` - Dependencies (~1 KB)
- ✅ `*.tex` - LaTeX source (~100 KB)
- ✅ `report_modern.pdf` - Report PDF (~300 KB)
- ✅ `.gitignore` - Git config

### ❌ KHÔNG NÊN PUSH (đã ignore)

- ❌ `*.pkl` - Feature files (126+ MB mỗi file!)
- ❌ `*.pth` - Model weights (100-500 MB mỗi file!)
- ❌ `*.jpg, *.png` - Dataset images (GBs!)
- ❌ `flickr*_images/` - Image folders (GBs!)
- ❌ `runs/`, `Dec*/` - TensorBoard logs (100+ MB)
- ❌ `__pycache__/` - Python cache
- ❌ `*.aux, *.log` - LaTeX temp files

---

## 🔧 Nếu vẫn gặp lỗi file quá lớn

### Nếu file đã vào Git history:

```powershell
# Kiểm tra các file lớn trong Git
git rev-list --objects --all | 
  Select-String -Pattern "features|\.pth|\.pkl" | 
  ForEach-Object { $_.ToString().Split()[1] }

# Nếu thấy file lớn, xóa .git và làm lại từ đầu (Bước 1)
Remove-Item -Recurse -Force .git
```

### Nếu muốn push một số files lớn:

**Sử dụng Git LFS** (Large File Storage):

```powershell
# Cài Git LFS
# Download từ: https://git-lfs.github.com/

# Khởi tạo Git LFS
git lfs install

# Track các file lớn
git lfs track "*.pkl"
git lfs track "*.pth"

# Add file .gitattributes
git add .gitattributes

# Commit và push
git add .
git commit -m "Add large files with Git LFS"
git push -u origin main
```

**Lưu ý**: Git LFS có giới hạn 1 GB free storage, sau đó phải trả phí.

---

## 📝 Checklist cuối cùng

Trước khi push, đảm bảo:

- [ ] Đã xóa thư mục `.git` cũ
- [ ] File `.gitignore` đã được cập nhật đầy đủ
- [ ] `git status` KHÔNG hiện các file lớn (*.pkl, *.pth, images)
- [ ] Tổng dung lượng repo < 300 MB
- [ ] Đã tạo Personal Access Token trên GitHub
- [ ] Remote URL đã đúng: `git remote -v`

---

## 🎯 Kết quả mong đợi

Sau khi hoàn thành, bạn sẽ có:

1. ✅ Repository trên GitHub với source code đầy đủ
2. ✅ README.md hiển thị đẹp với badges
3. ✅ Code có thể clone và chạy lại
4. ✅ Dung lượng repo < 50 MB (không tính LFS)

**Files không push**: Model weights và datasets sẽ được người dùng tự download từ Kaggle hoặc train lại.

---

## 💡 Lời khuyên

1. **Luôn kiểm tra .gitignore trước khi `git add .`**
2. **Sử dụng `git status` để xem files sẽ commit**
3. **Model weights và datasets: host riêng trên Google Drive/Kaggle**
4. **Chỉ push source code và documentation lên GitHub**

---

**Chúc bạn push thành công!** 🚀
