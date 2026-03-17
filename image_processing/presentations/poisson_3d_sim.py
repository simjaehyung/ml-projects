"""
Poisson Image Editing — 3D Brightness Surface Simulation
=========================================================
수식 흐름:
  E[f] = ∬|∇f - v|² dx dy  (에너지 최소화)
      → Euler-Lagrange 적용
      → Δf = div v  (Poisson 방정식)

시각화:
  X, Y 축  : 픽셀 위치
  Z 축     : 밝기(brightness) f
  소스     : 스페이드(♠) 모양, 내부 그레디언트 보유
  배경 A   : 부드러운 선형 그레디언트
  배경 B   : 고대비 정현파(sinusoidal) 패턴
  결과 A/B : Poisson 블렌딩 후 — 그레디언트 보존, 경계 seamless 연결
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import binary_erosion
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 0. 그리드 설정
# ============================================================
N = 100
rows_idx, cols_idx = np.mgrid[0:N, 0:N]
x_norm = cols_idx / (N - 1)   # [0, 1]  (좌→우)
y_norm = rows_idx / (N - 1)   # [0, 1]  (위→아래)

# ============================================================
# 1. 스페이드(♠) 마스크 생성
# ============================================================
def create_spade_mask(N):
    cx, cy = N / 2.0, N / 2.0
    r = N * 0.28

    dy = rows_idx.astype(float) - cy
    dx = cols_idx.astype(float) - cx

    # 왼쪽/오른쪽 원형 로브 (♠의 둥근 부분)
    left_lobe  = (dx + r * 0.44)**2 + (dy + r * 0.08)**2 <= (r * 0.53)**2
    right_lobe = (dx - r * 0.44)**2 + (dy + r * 0.08)**2 <= (r * 0.53)**2

    # 위쪽 삼각형 뾰족한 부분
    top_tri = (dy < -r * 0.06) & (np.abs(dx) < (-dy - r * 0.06) * 0.56)

    # 두 로브 사이 채우기 (중앙 하단 삼각)
    center_fill = (dy > r * 0.06) & (dy < r * 0.38) & \
                  (np.abs(dx) < (r * 0.38 - dy) * 0.72)

    # 수직 줄기
    stem = (np.abs(dx) < r * 0.14) & (dy > r * 0.32) & (dy < r * 0.66)

    # 밑받침 가로 막대
    base = (np.abs(dx) < r * 0.44) & (dy > r * 0.60) & (dy < r * 0.72)

    return left_lobe | right_lobe | top_tri | center_fill | stem | base


mask = create_spade_mask(N)

# ============================================================
# 2. 소스 이미지: 스페이드 내부 그레디언트
# ============================================================
cx, cy = N / 2.0, N / 2.0
r = N * 0.28
dy_c = rows_idx.astype(float) - cy
dx_c = cols_idx.astype(float) - cx

# 위쪽(0.88) → 아래쪽(0.22)으로 밝기 감소 + 가로 정현파 리플
t = np.clip((dy_c + r * 0.85) / (r * 1.75), 0.0, 1.0)
source_vals = 0.88 - 0.66 * t + 0.07 * np.sin(dx_c / r * np.pi * 1.6)
source = np.where(mask, source_vals, 0.0).astype(np.float64)

# 시각화용: 마스크 바깥은 바닥값으로
source_display = np.where(mask, source_vals, 0.04)

# ============================================================
# 3. 두 가지 배경 설정
# ============================================================
# 배경 A: 완만한 선형 그레디언트 (어두운 왼쪽 → 밝은 오른쪽)
bg_A = (0.06 + 0.30 * x_norm + 0.10 * y_norm).astype(np.float64)

# 배경 B: 고대비 정현파 패턴 (전혀 다른 밝기 지형)
bg_B = (0.52 + 0.32 * np.sin(x_norm * 2.6 * np.pi) *
        np.cos(y_norm * 1.9 * np.pi) + 0.07 * y_norm).astype(np.float64)

# ============================================================
# 4. Poisson 솔버  (벡터화 희소행렬)
#    구하는 것: Δf = Δsource (= div v)  안에서
#               f = target               경계에서
# ============================================================
def solve_poisson(src: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    N = src.shape[0]

    # 소스의 이산 라플라시안 (= div v)
    lap = np.zeros_like(src)
    lap[1:-1, 1:-1] = (src[2:, 1:-1] + src[:-2, 1:-1] +
                       src[1:-1, 2:] + src[1:-1, :-2] -
                       4.0 * src[1:-1, 1:-1])

    # 내부 픽셀 = 침식된 마스크 (경계를 제외한 순수 내부)
    interior = binary_erosion(mask)
    int_flat  = interior.ravel()
    int_idx   = np.where(int_flat)[0]   # 1D 평면 인덱스
    n         = len(int_idx)

    # 내부 픽셀 → 선형 시스템 인덱스 매핑
    idx_map = np.full(N * N, -1, dtype=np.int32)
    idx_map[int_idx] = np.arange(n, dtype=np.int32)

    # 희소 행렬 조립 (벡터화)
    row_lists, col_lists, dat_lists = [], [], []
    b = lap.ravel()[int_idx].copy()

    # 대각선 -4
    row_lists.append(np.arange(n))
    col_lists.append(np.arange(n))
    dat_lists.append(np.full(n, -4.0))

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        i2d = int_idx // N + di
        j2d = int_idx % N  + dj
        valid   = (i2d >= 0) & (i2d < N) & (j2d >= 0) & (j2d < N)
        nb_flat = i2d * N + j2d

        nb_v = nb_flat[valid]
        k_v  = np.where(valid)[0]
        is_int = idx_map[nb_v] >= 0

        # 이웃이 내부 → 행렬 계수 +1
        row_lists.append(k_v[is_int])
        col_lists.append(idx_map[nb_v[is_int]])
        dat_lists.append(np.ones(is_int.sum()))

        # 이웃이 경계(마스크 밖) → RHS에 배경값 추가
        b[k_v[~is_int]] -= tgt.ravel()[nb_v[~is_int]]

    A = csr_matrix(
        (np.concatenate(dat_lists),
         (np.concatenate(row_lists), np.concatenate(col_lists))),
        shape=(n, n)
    )
    sol = spsolve(A, b)

    result = tgt.copy()
    result.ravel()[int_idx] = sol
    return result


print("Poisson solving: Background A ...")
result_A = solve_poisson(source, bg_A, mask)
print("Poisson solving: Background B ...")
result_B = solve_poisson(source, bg_B, mask)
print("Done.")

# ============================================================
# 5. 그레디언트 크기 계산 (소스 vs 결과 비교용)
# ============================================================
def grad_mag(Z):
    gy, gx = np.gradient(Z)
    return np.sqrt(gx**2 + gy**2)

gm_source  = grad_mag(np.where(mask, source_vals, np.nan))
gm_result_A = grad_mag(result_A)
gm_result_B = grad_mag(result_B)

# ============================================================
# 6. 스페이드 경계선 (노란 점)
# ============================================================
interior_mask = binary_erosion(mask)
boundary      = mask.astype(bool) & ~interior_mask
bdy_r, bdy_c  = np.where(boundary)

def bdy_trace(Z):
    return go.Scatter3d(
        x=bdy_c / (N - 1),
        y=bdy_r / (N - 1),
        z=Z[bdy_r, bdy_c] + 0.018,
        mode='markers',
        marker=dict(size=2.2, color='#FFE44D', opacity=0.92),
        showlegend=False,
    )

# ============================================================
# 7. Plotly 3D 시각화 (2행 × 3열 + 1행 그레디언트 비교)
# ============================================================
LIGHT = dict(ambient=0.55, diffuse=0.88, specular=0.18, roughness=0.55)
LPOS  = dict(x=200, y=-100, z=300)

def surf(Z, cs, opacity=0.93, cmin=None, cmax=None):
    kw = dict(colorscale=cs, opacity=opacity, showscale=False,
              lighting=LIGHT, lightposition=LPOS,
              contours=dict(
                  z=dict(show=True, usecolormap=True,
                         highlightcolor='rgba(255,255,255,0.25)', project_z=False)
              ))
    if cmin is not None:
        kw['cmin'], kw['cmax'] = cmin, cmax
    return go.Surface(x=x_norm, y=y_norm, z=Z, **kw)


subplot_titles = [
    '① 원본 스페이드 — 내부 그레디언트<br>'
    '<sup>top(밝음 0.88) → bottom(어두움 0.22) + 가로 리플</sup>',

    '② Background A — Linear Gradient<br>'
    '<sup>f = 0.06 + 0.30·x + 0.10·y</sup>',

    '③ Result A: Δf = div v<br>'
    '<sup>스페이드 내부 그레디언트 보존 + BG-A 경계 seamless 연결</sup>',

    '① 원본 스페이드 — 동일 소스<br>'
    '<sup>완전히 같은 그레디언트 구조</sup>',

    '④ Background B — Sinusoidal Landscape<br>'
    '<sup>f = 0.52 + 0.32·sin(2.6πx)·cos(1.9πy)</sup>',

    '⑤ Result B: Δf = div v<br>'
    '<sup>다른 배경 → 다른 절댓값, 그러나 동일 그레디언트 구조</sup>',
]

fig = make_subplots(
    rows=2, cols=3,
    specs=[[{'type': 'surface'}] * 3] * 2,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.02,
    vertical_spacing=0.12,
)

# --- 서피스 + 경계 추가 ---
datasets = [
    (source_display, 'Hot'),
    (bg_A,           'Blues'),
    (result_A,       'Plasma'),
    (source_display, 'Hot'),
    (bg_B,           'Greens'),
    (result_B,       'Plasma'),
]
positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

for (row, col), (Z, cs) in zip(positions, datasets):
    fig.add_trace(surf(Z, cs, cmin=0.0, cmax=1.0), row=row, col=col)
    fig.add_trace(bdy_trace(Z), row=row, col=col)

# --- 화살표 어노테이션: 그레디언트 보존 강조 ---
# result 서피스 위에 텍스트 표시는 3D에서 어렵기 때문에
# 레이아웃 annotations 활용

# ============================================================
# 8. Scene(카메라/축) 설정
# ============================================================
ax_cfg = dict(
    backgroundcolor='#10102a',
    gridcolor='#282850',
    showbackground=True,
    tickfont=dict(size=8, color='#9999cc'),
    title_font=dict(size=9, color='#bbbbee'),
    zeroline=False,
)
scene_cfg = dict(
    xaxis=dict(title='x (위치)', **ax_cfg),
    yaxis=dict(title='y (위치)', **ax_cfg),
    zaxis=dict(title='밝기 f', range=[-0.05, 1.08], **ax_cfg),
    camera=dict(eye=dict(x=1.35, y=-1.35, z=1.05),
                up=dict(x=0, y=0, z=1)),
    bgcolor='#0a0a1f',
    aspectmode='manual',
    aspectratio=dict(x=1.0, y=1.0, z=0.68),
)

for i in range(1, 7):
    sname = 'scene' if i == 1 else f'scene{i}'
    fig.update_layout(**{sname: scene_cfg})

# ============================================================
# 9. 레이아웃 마무리
# ============================================================
for ann in fig.layout.annotations:
    ann.font = dict(size=10.5, color='#d0d0f8', family='Arial')

fig.update_layout(
    title=dict(
        text=(
            'Poisson Image Editing — 3D Brightness Surface Simulation<br>'
            '<sup>'
            'E[f] = ∬|∇f − v|² dxdy  →  Euler–Lagrange  →  <b>Δf = div v</b>'
            '&nbsp;&nbsp;|&nbsp;&nbsp;'
            '노란 점 = 스페이드 경계 (boundary condition: f = target)'
            '</sup>'
        ),
        font=dict(size=16, color='white', family='Arial'),
        x=0.5, xanchor='center', y=0.99,
    ),
    paper_bgcolor='#07071a',
    font=dict(color='#ccccee', family='Arial'),
    height=880,
    margin=dict(l=5, r=5, t=110, b=5),
    showlegend=False,
)

# ============================================================
# 10. 저장 및 실행
# ============================================================
out_path = r'c:\Users\jhsim\Erica261\M.L\projects\image_processing\presentations\poisson_3d_simulation.html'
fig.write_html(
    out_path,
    include_plotlyjs='cdn',
    full_html=True,
    config={'scrollZoom': True, 'displayModeBar': True},
)
print(f"\nSaved: {out_path}")
print("Open in browser for interactive 3D rotation/zoom.")
fig.show()
