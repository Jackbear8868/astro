# 在本機遠端檢視/操作伺服器上的 MUSE FITS Cube（QFitsView 與替代方案）

本文說明：如何在**本機**（你的筆電/桌機）檢視與操作**存放在這台遠端 Linux 伺服器**上的大型 MUSE
資料立方（Haro11，單檔約 7.6 GB），把運算留在伺服器、只把畫面帶回本機。你目前用 **VSCode Remote-SSH**
連線。以下所有環境結論都來自對本機（hostname `cv`）實際下指令的輸出。

---

## 結論（先看這裡，依推薦度排序）

| 排名 | 方案 | 一句話理由 | 是否需要 root | 是否需要 X11 |
|---|---|---|---|---|
| **1（首選）** | **jdaviz Cubeviz**（純瀏覽器） | 環境**已裝好** `jdaviz 5.0.2`，專為 IFU cube 設計，用伺服器算力，只需一個轉發埠 | 否 | **否** |
| 2 | Jupyter + `astropy`/`matplotlib` 自訂切片 | 最彈性、可腳本化；但要自己畫 | 否（見註） | 否 |
| 3 | **QFitsView + TigerVNC（使用者空間）+ 軟體 OpenGL** | 真的要用 QFitsView 時最穩的遠端顯示路徑 | 否 | 伺服器端虛擬桌面 |
| 4 | `ssh -Y` X11 forwarding 跑 QFitsView | 2D 切片可用；但 **3D cube 的 OpenGL 幾乎不能用** | 否 | 是（本機要有 X server） |

**一句話建議：** 你要的其實是「看 MUSE cube、把算力留在伺服器」——直接用**已經裝好的 jdaviz Cubeviz**，
不必碰 X11。只有在你明確需要 QFitsView 特定功能時，才走方案 3（VNC）。

---

## 一、本機環境探測結果（實際指令輸出）

| 項目 | 觀測值 | 指令 |
|---|---|---|
| OS | **Ubuntu 22.04.5 LTS**，kernel `6.8.0-124-generic`，x86_64，hostname `cv` | `cat /etc/os-release` / `uname -a` |
| 目前 SSH session 的顯示 | `DISPLAY` **空**、`WAYLAND_DISPLAY` **空**、`XDG_SESSION_TYPE=tty`（**無圖形 session**） | `echo $DISPLAY` … |
| 既有 X socket | `/tmp/.X11-unix/X0`，owner=`gdm` → 這是**機器實體螢幕的 GDM 登入畫面**，SSH session 無法使用 | `ls /tmp/.X11-unix` |
| sudo / root | **無免密 sudo**（`sudo: a password is required`）→ 視為**沒有 root** | `sudo -n true` |
| 使用者 / 群組 | `uid=1035(feather)`，groups=`feather`,`docker`（**不在 `video`/`render`**） | `id` |
| 已裝 X11 工具 | 只有 `xauth`、`xeyes`、`startx`；**沒有** `glxinfo`、`Xvfb`、`vncserver`、`x11vnc`、`vglrun` | `which …` |
| OpenGL 函式庫 | Mesa 與 NVIDIA 的 GLX/EGL **都在**；且有 `swrast_dri.so`、`zink_dri.so`（**軟體 OpenGL 可用**） | `ldconfig -p` / `ls /usr/lib/x86_64-linux-gnu/dri` |
| GPU | **NVIDIA RTX 4090 24 GB**，driver `550.144.03`；`/dev/dri/card0`、`renderD128` 存在 | `nvidia-smi -L` / `ls /dev/dri` |
| **GPU 存取權限** | `/dev/dri` 的 ACL 只給 `gdm` 與群組 `video`/`render`，`other::---`。**`feather` 不在其中 → 你的帳號無法直接使用這顆 GPU 做 headless 3D 硬體加速** | `getfacl /dev/dri/*` |
| astro conda 環境 | `astropy 8.0.0`、Python `3.12.13`、**`jdaviz 5.0.2`（已裝）**、`solara 1.57.6`、`jupyter_server 2.20.0`；**`jupyterlab`/`notebook`/`voila` 未裝** | `conda run -n astro …` |
| 其他檢視器 | **沒有** `ds9`/`js9`/`pyds9` | `which ds9 js9` |
| sshd | `X11Forwarding yes`（伺服器允許 X11 轉發） | `grep -i x11 /etc/ssh/sshd_config` |
| MUSE 資料 | `/local/feather/workspace/astro/data/`：`Haro11_wsky.fits` 7.6 GB、`Haro11_nosky.fits` 7.6 GB、`Haro11_WFM_MUSE_archive.fits` 7.0 GB 等 | `find … -iname '*.fits'` |

**兩個關鍵結論：**
1. 這台機器是**無頭（headless）**的：你的 SSH session 沒有任何 X server 可用，那個 `X0` 是實體螢幕的
   GDM 登入畫面、不屬於你。任何 GUI 都必須**自己起一個顯示**（本機 X server 或伺服器端虛擬桌面）。
2. 這台機器**有一顆 RTX 4090，但你的帳號目前碰不到它**（不在 `render`/`video` 群組）。因此 GPU 硬體加速的
   遠端 3D（VirtualGL+EGL）目前**不可行**，除非請系統管理員把 `feather` 加入 `render` 群組。好消息是：
   **jdaviz / astropy 這條路根本不需要 GPU 或 X11**。

---

## 二、為什麼「VSCode Remote-SSH 本身」開不了 GUI

VSCode Remote-SSH 在伺服器上跑的是一個 **headless 的 VSCode server**，它只轉發：**TCP 埠**、**終端機**、
**檔案系統**。它**不是** X server，也**不含**任何顯示伺服器。所以：

- 你在 VSCode 的整合終端機打 `QFitsView`，程式會找 `DISPLAY`，發現是空的 → **直接報錯結束**。
- VSCode 能做的、而且非常有用的事，是**埠轉發（Port Forwarding）**：任何在伺服器 `localhost:PORT`
  監聽的服務，都能被 VSCode 自動或手動轉發到你本機的 `localhost:PORT`，用本機瀏覽器打開。

**這正是方案 1/2 的運作基礎**：在伺服器跑一個「網頁版」檢視器，VSCode 把埠帶回本機瀏覽器。GUI 的問題就
繞過去了。

---

## 三、各種「把畫面帶回本機」的傳輸方案（含各自的唯一致命點）

### (A) `ssh -X` / `ssh -Y` X11 forwarding
把伺服器上 X 應用的視窗，透過 SSH 通道畫到**你本機的 X server**上（Linux 原生即有；macOS 裝
**XQuartz**；Windows 裝 **VcXsrv / X410 / MobaXterm**）。可與 VSCode 並用：VSCode 負責編輯，另開一個
`ssh -Y` 終端跑 GUI。

```bash
# 本機（Linux/macOS(XQuartz)）：
ssh -Y feather@<server>
# 進去後（2D 影像檢視大致可用）：
DISPLAY 已由 ssh 設好，直接跑你的 X 程式
```

- **唯一致命點：OpenGL。** QFitsView 的 **3D cube / volume 渲染走 OpenGL**，而 X11 轉發只能做
  **indirect GLX**，上限僅 OpenGL 1.4、常常直接失敗或慢到不能用（現代 Qt 需要 GL 2.0+）。也就是說
  **2D 切片勉強能看，3D cube 幾乎不能用**。`LIBGL_ALWAYS_INDIRECT=1`、`xset` 之類只能治標、無法讓現代
  GL 應用順跑。大檔互動延遲也高（每次重繪都要走網路）。

### (B) VNC（TigerVNC / TurboVNC）+ 伺服器端虛擬桌面
在伺服器上開一個**虛擬 X 桌面（Xvnc）**，把它的畫面壓縮成 VNC 串流，透過 SSH/VSCode 轉發 VNC 埠
（5901）到本機，用本機的 VNC viewer 觀看。GL 應用在伺服器端渲染好、只把「畫好的畫面」傳回來，互動比
X11 轉發穩很多。

- 若要 **GPU 硬體加速** GL，需搭 **VirtualGL**——但 VirtualGL 需要能存取 GPU 的 render node。
  **本機這顆 4090 你的帳號沒權限（見上）**，所以只能走**軟體 OpenGL（Mesa llvmpipe，`swrast_dri.so`
  已在）**：正確、可用，但 3D 較慢；2D 完全順。
- **唯一致命點：** 伺服器上沒有預裝 `vncserver`，且你沒有 root。→ 需**使用者空間安裝**（下方方案 3 給了
  免 root 的 TigerVNC tarball 作法）。

### (C) NoMachine / X2Go
桌面串流方案。**X2Go** 免 root 較容易，但**對 OpenGL 應用支援不佳**（和 X11 轉發類似的 GL 問題）。
**NoMachine** 的 GL 處理較好、體驗佳，但通常**需要在伺服器逐機安裝**（多半要 root），此環境不適用。
→ 這裡**不推薦**。

### (D) 純軟體渲染 fallback（完全無 GPU/無實體顯示）
`Xvfb`（虛擬 framebuffer）或 `Xvnc` + Mesa **llvmpipe/OSMesa** 軟體 GL。慢，但**完全 headless 可用、
不需任何實體螢幕或 GPU 權限**。本機 `swrast_dri.so`、`zink_dri.so` 已在，具備此能力。方案 3 實際就是走
這條（Xvnc + `LIBGL_ALWAYS_SOFTWARE=1`）。

---

## 四、QFitsView 專屬說明（來自官方 www.mpe.mpg.de/~ott/QFitsView/）

- **版本 / 下載：** 目前 **QFitsView 4.3（2025-03）**。Linux 只提供**單一個執行檔 binary**（非 AppImage、
  非 static 說明），下載後 `chmod a+x` 即可執行。原始碼可下載但官方明言**相依函式庫多、不好編**、不提供
  編譯支援。
- **免 root：** 因為就是一個可執行檔，**不需要 root**，直接下載到家目錄 `chmod +x` 就能跑——**安裝不是問題，
  「把 GUI 顯示到本機」才是問題。**
- **相依：** 以 **Qt** 撰寫；**3D cube / volume 檢視使用 OpenGL**（這正是遠端顯示的痛點，見方案 A）。2D
  影像/切片檢視不吃 GL，遠端相對容易。
- 因此若你在遠端用 QFitsView，**務必用 VNC（方案 3）而不是 `ssh -X`**；並用**軟體 OpenGL**（你的帳號沒
  GPU 權限）。

下載（在伺服器上）：
```bash
mkdir -p ~/apps && cd ~/apps
curl -fLO https://www.mpe.mpg.de/~ott/QFitsView/download/QFitsView_4.3
chmod a+x QFitsView_4.3
```

---

## 五、推薦作法（含可直接複製的指令）

### ★ 方案 1（首選）：jdaviz Cubeviz — 純瀏覽器、用伺服器算力、零 X11

環境**已經裝好** `jdaviz 5.0.2` + `solara`，這是 astropy 生態、專門為 IFU/MUSE data cube 設計的網頁
檢視器：影像切片、逐 spaxel 取譜、光譜/波長瀏覽、collapse、subset 等都有。它在伺服器起一個網頁服務，
你用**本機瀏覽器**看，運算全在伺服器。

**步驟 1｜在 VSCode 的整合終端機（伺服器端）啟動：**
```bash
conda run -n astro jdaviz --layout cubeviz \
  -fp /local/feather/workspace/astro/data/Haro11_nosky.fits \
  --host 127.0.0.1 --port 8765
```
- 啟動後終端會印出網址（例如 `http://127.0.0.1:8765`）。在 headless 機器上它嘗試自動開瀏覽器會靜默失敗，
  這**正常**——我們用轉發埠自己開。
- `--layout cubeviz` 在 5.0 標記為 deprecated 但仍可用；也可不加 `--layout` 直接 `jdaviz` 進通用介面，
  再從 UI 載入 cube。

**步驟 2｜把 8765 埠帶回本機：**
- **VSCode 方式（最簡單）：** VSCode 偵測到伺服器在 `localhost:8765` 監聽時，會在 **PORTS** 面板自動列出；
  沒自動出現就手動 **Forward a Port → 8765**，再點該列的地球圖示用本機瀏覽器開啟。
- **純 SSH 方式（等效）：**
  ```bash
  ssh -N -L 8765:localhost:8765 feather@<server>
  # 然後本機瀏覽器開 http://localhost:8765
  ```

**唯一注意點（記憶體）：** MUSE cube 約 7.6 GB，Cubeviz 會把資料載入記憶體，載入與互動會偏重。若卡頓，
先在伺服器用 `astropy` 裁一個波長子立方或空間子區來看：
```python
# conda run -n astro python
from astropy.io import fits
h = fits.open('/local/feather/workspace/astro/data/Haro11_nosky.fits')
# 例如只取 Halpha 附近若干波長層，寫出小 cube，再丟給 jdaviz
```

### 方案 2：Jupyter Notebook + astropy/matplotlib（最彈性）
適合你已在做的 sky subtraction 分析：直接在 notebook 裡切片、畫圖、疊 mask。`astropy 8.0.0`、`ipykernel`
已在；但 **`jupyterlab`/`notebook` 尚未安裝**。因為這是**共用的 astro 環境**，建議**另建一個環境或用
`pip install --user`**，避免動到共用環境：
```bash
# 建議：獨立環境（不動共用 astro）
conda create -n viz python=3.12 jupyterlab astropy matplotlib -y
conda run -n viz jupyter lab --no-browser --ServerApp.ip=127.0.0.1 --port 8888
# 然後同樣把 8888 埠用 VSCode PORTS 或 ssh -L 帶回本機
```
（若你偏好，jdaviz 也能**內嵌在 notebook** 裡跑，`from jdaviz import Cubeviz`。）

### 方案 3（真的要 QFitsView）：TigerVNC 使用者空間 + 軟體 OpenGL + 埠轉發
這條路免 root、最穩地把 QFitsView（含 3D）顯示到本機。GPU 你的帳號沒權限，所以用**軟體 OpenGL**。

**步驟 1｜使用者空間安裝 TigerVNC（免 root，解壓到家目錄）：**
```bash
mkdir -p ~/tigervnc && cd ~/tigervnc
# 從 TigerVNC 官方 release 抓 x86_64 通用 tarball（版本號以官方最新為準）
curl -fLO https://github.com/TigerVNC/tigervnc/releases/download/v1.14.1/tigervnc-1.14.1.x86_64.tar.gz
tar xzf tigervnc-1.14.1.x86_64.tar.gz --strip-components=1
export PATH="$HOME/tigervnc/usr/bin:$PATH"
```

**步驟 2｜設定密碼與精簡 xstartup（沒有預裝的桌面/WM，直接跑 QFitsView）：**
```bash
mkdir -p ~/.vnc
vncpasswd                      # 設一個 VNC 連線密碼
cat > ~/.vnc/xstartup <<'EOF'
#!/bin/sh
export LIBGL_ALWAYS_SOFTWARE=1   # 用 Mesa llvmpipe 軟體 GL（你的帳號無 GPU 權限）
exec ~/apps/QFitsView_4.3        # 直接把 QFitsView 當作 session 主程式
EOF
chmod +x ~/.vnc/xstartup
```
（若想要視窗管理器讓視窗可移動/縮放，可再 `conda install -n viz -c conda-forge fluxbox`，把 xstartup
改成先 `fluxbox &` 再啟動 QFitsView。）

**步驟 3｜啟動只綁 localhost 的 VNC 桌面：**
```bash
vncserver :1 -localhost -geometry 1600x1000 -depth 24
# → 服務在伺服器 127.0.0.1:5901（display :1 = 5900+1）
```

**步驟 4｜轉發 5901 到本機並用 VNC viewer 連：**
```bash
# 純 SSH：
ssh -N -L 5901:localhost:5901 feather@<server>
# 或用 VSCode PORTS 面板 Forward a Port → 5901
# 本機用任一 VNC client（TigerVNC Viewer / RealVNC / macOS 內建）連 localhost:5901
```

**唯一致命點：** 3D volume 用軟體 GL 會**偏慢**（2D 切片順）。若要快，唯一解是請**系統管理員把 `feather`
加入 `render`（與 `video`）群組**，屆時可改用 **VirtualGL 的 EGL 後端**（`vglrun -d egl QFitsView_4.3`）
直接吃 RTX 4090 做 headless 硬體加速，完全不需實體螢幕。

用完記得關閉：`vncserver -kill :1`。

---

## 六、關於 GPU 與 `docker` 群組（進階備註）

- **目前你的帳號無法直接用這顆 RTX 4090**（`/dev/dri` 的 ACL 只給 `gdm` 與 `video`/`render` 群組）。若你的
  工作流需要 GPU 加速的遠端 3D，最乾淨的作法是**請管理員把 `feather` 加入 `render` 群組**，然後走
  「VNC + VirtualGL(EGL)」。
- 你在 `docker` 群組中（等同 root 權限的逃生門）：理論上可在容器內以 root 安裝任意套件、並用
  `--gpus all` 取得 GPU，再於容器內跑 VNC/QFitsView。但這實際上等於繞過主機權限管制、且設定較重，**除非
  必要不建議**——jdaviz 這條路完全不需要它。

---

## 七、一句話收尾

**先用方案 1（jdaviz Cubeviz，已裝好、零 X11、用伺服器算力、VSCode 轉發一個埠即可）。** 需要腳本化分析時
搭方案 2 的 Jupyter。只有在你明確需要 QFitsView 的特定功能時，才走方案 3（使用者空間 TigerVNC + 軟體
OpenGL）；並記得——若要 GPU 加速，得先請管理員把你加進 `render` 群組。

---

### 參考來源
- QFitsView 官方站（Thomas Ott, MPE）：<https://www.mpe.mpg.de/~ott/QFitsView/>
- QFitsView Ubuntu manpage：<https://manpages.ubuntu.com/manpages/jammy/man1/QFitsView.1.html>
- VirtualGL（遠端 OpenGL 加速）：<https://wiki.archlinux.org/title/VirtualGL>
- OpenGL over remote X11 / indirect GLX 限制：<https://evpo.wordpress.com/2017/03/04/opengl-hardware-acceleration-through-remote-x11-ssh-connection/>
- jdaviz（Cubeviz，astropy 生態）：<https://jdaviz.readthedocs.io/>
