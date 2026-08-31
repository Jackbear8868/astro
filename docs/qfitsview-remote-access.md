# Viewing and working with the server's MUSE FITS cubes from your own machine (QFitsView and the alternatives)

This note explains how to view and work with the large MUSE data cubes **stored on this remote
Linux server** (Haro11, about 7.6 GB per file) from **your own machine** — your laptop or
desktop — leaving the computation on the server and bringing back only the picture. You currently
connect with **VSCode Remote-SSH**. Every conclusion about the environment below comes from
commands actually run on this machine (hostname `cv`).

---

## The conclusion (read this first, in order of preference)

| Rank | Option | The reason in one line | Needs root | Needs X11 |
|---|---|---|---|---|
| **1 (first choice)** | **jdaviz Cubeviz** (browser only) | `jdaviz 5.0.2` is **already installed**, it is built for IFU cubes, it runs on the server's compute, and it needs one forwarded port | No | **No** |
| 2 | Jupyter + `astropy`/`matplotlib` slices of your own | The most flexible, and scriptable; but you draw everything yourself | No (see note) | No |
| 3 | **QFitsView + TigerVNC (user space) + software OpenGL** | The most dependable remote-display route when you really do need QFitsView | No | A virtual desktop on the server |
| 4 | `ssh -Y` X11 forwarding running QFitsView | 2D slices are usable, but **the 3D cube's OpenGL is all but unusable** | No | Yes (your own machine needs an X server) |

**The recommendation in one line:** what you actually want is to look at a MUSE cube while leaving
the compute on the server — so use **the jdaviz Cubeviz that is already installed** and never touch
X11. Take option 3 (VNC) only when you specifically need something QFitsView does.

---

## I. What the survey of this machine found (actual command output)

| Item | Observed | Command |
|---|---|---|
| OS | **Ubuntu 22.04.5 LTS**, kernel `6.8.0-124-generic`, x86_64, hostname `cv` | `cat /etc/os-release` / `uname -a` |
| The display of the current SSH session | `DISPLAY` **empty**, `WAYLAND_DISPLAY` **empty**, `XDG_SESSION_TYPE=tty` (**no graphical session**) | `echo $DISPLAY` … |
| The X socket that exists | `/tmp/.X11-unix/X0`, owner=`gdm` → this is **the GDM login screen on the machine's physical monitor**, and an SSH session cannot use it | `ls /tmp/.X11-unix` |
| sudo / root | **No passwordless sudo** (`sudo: a password is required`) → treat this as **having no root** | `sudo -n true` |
| User / groups | `uid=1035(feather)`, groups=`feather`,`docker` (**not in `video`/`render`**) | `id` |
| X11 tools installed | Only `xauth`, `xeyes`, `startx`; **no** `glxinfo`, `Xvfb`, `vncserver`, `x11vnc`, `vglrun` | `which …` |
| OpenGL libraries | Mesa's and NVIDIA's GLX/EGL are **both present**, and `swrast_dri.so` and `zink_dri.so` are there too (**software OpenGL is available**) | `ldconfig -p` / `ls /usr/lib/x86_64-linux-gnu/dri` |
| GPU | **NVIDIA RTX 4090 24 GB**, driver `550.144.03`; `/dev/dri/card0` and `renderD128` exist | `nvidia-smi -L` / `ls /dev/dri` |
| **GPU access rights** | The ACL on `/dev/dri` grants access only to `gdm` and to the groups `video`/`render`, with `other::---`. **`feather` is in none of them → your account cannot use this GPU directly for headless 3D hardware acceleration** | `getfacl /dev/dri/*` |
| The astro conda environment | `astropy 8.0.0`, Python `3.12.13`, **`jdaviz 5.0.2` (already installed)**, `solara 1.57.6`, `jupyter_server 2.20.0`; **`jupyterlab`/`notebook`/`voila` are not installed** | `conda run -n astro …` |
| Other viewers | **No** `ds9`/`js9`/`pyds9` | `which ds9 js9` |
| sshd | `X11Forwarding yes` (the server does permit X11 forwarding) | `grep -i x11 /etc/ssh/sshd_config` |
| MUSE data | In `/local/feather/workspace/sky-subtraction/data/`: `Haro11_wsky.fits` 7.6 GB, `Haro11_nosky.fits` 7.6 GB, `Haro11_WFM_MUSE_archive.fits` 7.0 GB, among others | `find … -iname '*.fits'` |

**Two conclusions that matter:**
1. This machine is **headless**: your SSH session has no X server available to it at all, and that
   `X0` is the GDM login screen of the physical monitor, which is not yours. Any GUI must
   **bring up a display of its own** — either an X server on your machine or a virtual desktop on
   the server.
2. This machine **has an RTX 4090, but your account cannot currently touch it** (you are not in the
   `render`/`video` groups). GPU-accelerated remote 3D (VirtualGL+EGL) is therefore **not possible**
   at present, unless you ask the system administrator to add `feather` to the `render` group. The
   good news: **the jdaviz / astropy route needs neither a GPU nor X11.**

---

## II. Why VSCode Remote-SSH by itself cannot open a GUI

What VSCode Remote-SSH runs on the server is a **headless VSCode server**, and it forwards three
things only: **TCP ports**, **the terminal**, and **the file system**. It is **not** an X server, and
it **contains** no display server. So:

- If you type `QFitsView` in VSCode's integrated terminal, the program looks for `DISPLAY`, finds it
  empty, and **exits with an error straight away**.
- What VSCode can do, and this is genuinely useful, is **port forwarding**: any service listening on
  `localhost:PORT` on the server can be forwarded, automatically or by hand, to `localhost:PORT` on
  your own machine and opened in your own browser.

**This is exactly what options 1 and 2 rest on**: run a browser-based viewer on the server, and let
VSCode carry the port back to your local browser. The GUI problem is then simply sidestepped.

---

## III. The ways of getting the picture back to your machine (each with its one fatal flaw)

### (A) `ssh -X` / `ssh -Y` X11 forwarding

The window of an X application on the server is drawn, through the SSH tunnel, on **the X server on
your own machine** (Linux has one natively; on macOS install **XQuartz**, on Windows install
**VcXsrv / X410 / MobaXterm**). It coexists with VSCode: VSCode does the editing, and a separate
`ssh -Y` terminal runs the GUI.

```bash
# On your own machine (Linux/macOS(XQuartz)):
ssh -Y feather@<server>
# Once you are in (2D image viewing is broadly usable):
DISPLAY has already been set by ssh, so just run your X program
```

- **The one fatal flaw: OpenGL.** QFitsView's **3D cube / volume rendering goes through OpenGL**,
  and X11 forwarding can only do **indirect GLX**, which tops out at OpenGL 1.4 and often either
  fails outright or is too slow to use (modern Qt needs GL 2.0+). In other words, **2D slices are
  just about viewable, and the 3D cube is all but unusable**. `LIBGL_ALWAYS_INDIRECT=1`, `xset` and
  the like treat the symptom and cannot make a modern GL application run smoothly. Interaction with
  large files is also laggy, since every redraw crosses the network.

### (B) VNC (TigerVNC / TurboVNC) + a virtual desktop on the server

Open a **virtual X desktop (Xvnc)** on the server, compress its screen into a VNC stream, forward the
VNC port (5901) to your machine over SSH or VSCode, and watch it with a local VNC viewer. The GL
application renders on the server and only the finished picture comes back, which makes interaction
far steadier than X11 forwarding.

- For **GPU hardware acceleration** of GL you would pair this with **VirtualGL** — but VirtualGL needs
  access to the GPU's render node. **Your account has no permission on this machine's 4090 (see
  above)**, so the only route open is **software OpenGL (Mesa llvmpipe; `swrast_dri.so` is already
  present)**: correct and usable, but slow in 3D, while 2D is perfectly smooth.
- **The one fatal flaw:** the server has no `vncserver` preinstalled, and you have no root. → You need
  a **user-space installation** (option 3 below gives the root-free TigerVNC tarball recipe).

### (C) NoMachine / X2Go

Desktop-streaming solutions. **X2Go** is the easier one to install without root, but **its support for
OpenGL applications is poor** (the same class of GL problem as X11 forwarding). **NoMachine** handles
GL better and is pleasant to use, but it usually **has to be installed on the server, machine by
machine** (which mostly means root), so it does not fit this environment. → **Not recommended** here.

### (D) Pure software-rendering fallback (no GPU and no physical display at all)

`Xvfb` (a virtual framebuffer), or `Xvnc` plus Mesa **llvmpipe/OSMesa** software GL. Slow, but
**fully usable headless, requiring neither a physical monitor nor GPU permissions**. This machine
already has `swrast_dri.so` and `zink_dri.so`, so the capability is there. Option 3 is in fact this
route (Xvnc + `LIBGL_ALWAYS_SOFTWARE=1`).

---

## IV. Notes specific to QFitsView (from the official site, www.mpe.mpg.de/~ott/QFitsView/)

- **Version / download:** the current release is **QFitsView 4.3 (2025-03)**. For Linux it is
  distributed as **a single executable binary** (not an AppImage, and with no statement that it is
  static); download it, `chmod a+x`, and it runs. The source can be downloaded, but the authors say
  plainly that it **has many dependencies and is not easy to build**, and they offer no build support.
- **No root needed:** because it is just an executable, it **needs no root** — download it into your
  home directory, `chmod +x`, and run it. **Installing it is not the problem; getting the GUI onto
  your own screen is the problem.**
- **Dependencies:** it is written in **Qt**, and its **3D cube / volume view uses OpenGL** — which is
  precisely the sore point for remote display, see option A. The 2D image/slice view does not use GL,
  so it is comparatively easy to run remotely.
- Hence, if you use QFitsView remotely, **be sure to use VNC (option 3) and not `ssh -X`**, and use
  **software OpenGL** (your account has no GPU permissions).

Downloading (on the server):
```bash
mkdir -p ~/apps && cd ~/apps
curl -fLO https://www.mpe.mpg.de/~ott/QFitsView/download/QFitsView_4.3
chmod a+x QFitsView_4.3
```

---

## V. The recommended recipes (with commands you can copy straight out)

### ★ Option 1 (first choice): jdaviz Cubeviz — browser only, server compute, zero X11

The environment **already has** `jdaviz 5.0.2` + `solara` installed. This is the astropy ecosystem's
web viewer built specifically for IFU/MUSE data cubes: image slices, per-spaxel spectrum extraction,
spectrum/wavelength browsing, collapse, subsets and so on are all there. It starts a web service on
the server, you view it in **your own browser**, and all the computation stays on the server.

**Step 1 | Start it from VSCode's integrated terminal (i.e. on the server):**
```bash
conda run -n astro jdaviz --layout cubeviz \
  -fp /local/feather/workspace/sky-subtraction/data/Haro11_nosky.fits \
  --host 127.0.0.1 --port 8765
```
- Once started, the terminal prints a URL (for example `http://127.0.0.1:8765`). On a headless
  machine its attempt to open a browser automatically fails silently, which is **normal** — we open
  it ourselves through the forwarded port.
- `--layout cubeviz` is marked deprecated in 5.0 but still works; you can equally drop `--layout`,
  run plain `jdaviz` to get the general interface, and load the cube from the UI.

**Step 2 | Bring port 8765 back to your machine:**
- **The VSCode way (simplest):** when VSCode notices the server listening on `localhost:8765` it lists
  it automatically in the **PORTS** panel; if it does not appear on its own, do
  **Forward a Port → 8765** by hand, then click the globe icon on that row to open it in your own
  browser.
- **The pure SSH way (equivalent):**
  ```bash
  ssh -N -L 8765:localhost:8765 feather@<server>
  # then open http://localhost:8765 in your own browser
  ```

**The one thing to watch (memory):** a MUSE cube is about 7.6 GB, Cubeviz loads the data into memory,
and both loading and interacting will be heavy. If it stutters, first use `astropy` on the server to
cut out a wavelength sub-cube or a spatial sub-region to look at:
```python
# conda run -n astro python
from astropy.io import fits
h = fits.open('/local/feather/workspace/sky-subtraction/data/Haro11_nosky.fits')
# for example, take just the wavelength layers near Halpha, write out a small cube, and hand that to jdaviz
```

### Option 2: Jupyter Notebook + astropy/matplotlib (the most flexible)
This suits the sky subtraction analysis you are already doing: slice, plot and overlay masks directly
in a notebook. `astropy 8.0.0` and `ipykernel` are already there, but **`jupyterlab`/`notebook` are
not yet installed**. Since this is **a shared astro environment**, it is better to **create a separate
environment or use `pip install --user`**, so that the shared one is left alone:
```bash
# Recommended: a separate environment (leaves the shared astro alone)
conda create -n viz python=3.12 jupyterlab astropy matplotlib -y
conda run -n viz jupyter lab --no-browser --ServerApp.ip=127.0.0.1 --port 8888
# then bring port 8888 back the same way, with the VSCode PORTS panel or ssh -L
```
(If you prefer, jdaviz can also run **embedded in a notebook**, with `from jdaviz import Cubeviz`.)

### Option 3 (when you really do need QFitsView): user-space TigerVNC + software OpenGL + port forwarding
This is the route that needs no root and most dependably puts QFitsView, 3D included, on your own
screen. Your account has no GPU permission, so it uses **software OpenGL**.

**Step 1 | Install TigerVNC in user space (no root; unpack into your home directory):**
```bash
mkdir -p ~/tigervnc && cd ~/tigervnc
# fetch the generic x86_64 tarball from TigerVNC's official releases (use whatever version is current)
curl -fLO https://github.com/TigerVNC/tigervnc/releases/download/v1.14.1/tigervnc-1.14.1.x86_64.tar.gz
tar xzf tigervnc-1.14.1.x86_64.tar.gz --strip-components=1
export PATH="$HOME/tigervnc/usr/bin:$PATH"
```

**Step 2 | Set a password and a minimal xstartup (there is no desktop or WM preinstalled, so run QFitsView directly):**
```bash
mkdir -p ~/.vnc
vncpasswd                      # choose a password for the VNC connection
cat > ~/.vnc/xstartup <<'EOF'
#!/bin/sh
export LIBGL_ALWAYS_SOFTWARE=1   # use Mesa llvmpipe software GL (your account has no GPU permission)
exec ~/apps/QFitsView_4.3        # make QFitsView itself the session's main program
EOF
chmod +x ~/.vnc/xstartup
```
(If you want a window manager so that windows can be moved and resized, add
`conda install -n viz -c conda-forge fluxbox` and change xstartup to run `fluxbox &` before starting
QFitsView.)

**Step 3 | Start a VNC desktop bound to localhost only:**
```bash
vncserver :1 -localhost -geometry 1600x1000 -depth 24
# → the service is on the server at 127.0.0.1:5901 (display :1 = 5900+1)
```

**Step 4 | Forward 5901 to your machine and connect with a VNC viewer:**
```bash
# Pure SSH:
ssh -N -L 5901:localhost:5901 feather@<server>
# or use the VSCode PORTS panel, Forward a Port → 5901
# on your own machine, connect to localhost:5901 with any VNC client (TigerVNC Viewer / RealVNC / the one built into macOS)
```

**The one fatal flaw:** 3D volume rendering under software GL is **on the slow side** (2D slices are
smooth). The only way to make it fast is to ask the **system administrator to add `feather` to the
`render` (and `video`) group**; at that point you could switch to **VirtualGL's EGL backend**
(`vglrun -d egl QFitsView_4.3`) and drive the RTX 4090 directly for headless hardware acceleration,
with no physical monitor involved at all.

Remember to shut it down when you are done: `vncserver -kill :1`.

---

## VI. On the GPU and the `docker` group (an advanced note)

- **Your account cannot currently use this RTX 4090 directly** (the ACL on `/dev/dri` grants access
  only to `gdm` and the `video`/`render` groups). If your workflow does need GPU-accelerated remote
  3D, the cleanest thing to do is **ask the administrator to add `feather` to the `render` group** and
  then take the "VNC + VirtualGL(EGL)" route.
- You are in the `docker` group, which is an escape hatch equivalent to root: in principle you could
  install anything as root inside a container, get at the GPU with `--gpus all`, and run
  VNC/QFitsView inside the container. But this amounts to going around the host's permission controls
  and is heavy to set up, so **it is not advisable unless you have to** — and the jdaviz route does
  not need it at all.

---

## VII. The closing line

**Start with option 1 (jdaviz Cubeviz: already installed, zero X11, server compute, one port forwarded
by VSCode and that is all).** When you need scripted analysis, add option 2's Jupyter. Go to option 3
(user-space TigerVNC + software OpenGL) only when you specifically need something QFitsView does — and
remember that GPU acceleration means first asking the administrator to put you in the `render` group.

---

### Sources
- The official QFitsView site (Thomas Ott, MPE): <https://www.mpe.mpg.de/~ott/QFitsView/>
- The QFitsView Ubuntu manpage: <https://manpages.ubuntu.com/manpages/jammy/man1/QFitsView.1.html>
- VirtualGL (remote OpenGL acceleration): <https://wiki.archlinux.org/title/VirtualGL>
- The limits of OpenGL over remote X11 / indirect GLX: <https://evpo.wordpress.com/2017/03/04/opengl-hardware-acceleration-through-remote-x11-ssh-connection/>
- jdaviz (Cubeviz, in the astropy ecosystem): <https://jdaviz.readthedocs.io/>
