"""
PneumoScan – Détection de pneumonie sur radiographies thoraciques
Projet académique – Yousra Erraki (EMSI)

Modèle : VGG16 (Transfer Learning), entraîné sur le dataset Kaggle « Chest X-Ray Images (Pneumonia) ».

Lancer :  streamlit run app.py
Placer le modèle (.h5) dans le même dossier que app.py.
Les performances sont calculées directement sur le dossier de test du dataset (onglet « Le modèle »).
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(page_title="PneumoScan", page_icon="🫁", layout="wide")

HERE = Path(__file__).parent
MODEL_FILES = ["modelVGG16_FineTuning.h5", "modelVGG_Pneumonia_TL.h5"]
IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA"]
EXTS = {".jpg", ".jpeg", ".png"}

# ─────────────────────────── Style ───────────────────────────
BG, PANEL, BORDER, TEXT, MUTED = "#0B1220", "#111A2E", "#1F2A44", "#E6EDF7", "#8A9BB8"
CYAN, GREEN, RED = "#22D3EE", "#34D399", "#F87171"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #12305a 0%, {BG} 55%) fixed; color: {TEXT}; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
[data-testid="stSidebar"] {{ background: #070C17; border-right: 1px solid {BORDER}; }}

.hero {{ display:flex; align-items:center; gap:22px; padding: 26px 30px; border-radius: 20px;
  background: linear-gradient(120deg, rgba(34,211,238,.14), rgba(99,102,241,.10));
  border: 1px solid {BORDER}; margin-bottom: 22px; }}
.hero .icon {{ font-size: 54px; }}
.hero h1 {{ margin: 0; font-size: 34px; font-weight: 800; letter-spacing: -.5px; color: {TEXT}; }}
.hero h1 span {{ color: {CYAN}; }}
.hero p {{ margin: 6px 0 0; color: {MUTED}; font-size: 15.5px; }}
.pill {{ display:inline-block; font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: {CYAN}; border: 1px solid rgba(34,211,238,.4); padding: 3px 10px; border-radius: 999px; margin-bottom: 8px; }}

.panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 18px; padding: 20px 22px; }}
.label {{ color: {MUTED}; font-size: 12.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; }}
.kpi .v {{ font-size: 28px; font-weight: 800; margin-top: 4px; }}
.kpi .s {{ color: {MUTED}; font-size: 13px; }}
.result {{ border-radius: 18px; padding: 22px 24px; margin-top: 4px; }}
.result h2 {{ margin: 0; font-size: 26px; font-weight: 800; }}
.result p {{ margin: 6px 0 0; opacity: .9; }}
.bar {{ height: 10px; background: #1B2540; border-radius: 999px; overflow: hidden; margin: 6px 0 14px; }}
.bar > div {{ height: 100%; border-radius: 999px; }}
.disclaimer {{ color: {MUTED}; font-size: 12.5px; border-left: 3px solid #F59E0B; padding: 8px 12px;
  background: rgba(245,158,11,.06); border-radius: 6px; margin-top: 14px; }}

[data-testid="stFileUploaderDropzone"] {{ background: {PANEL}; border: 1.5px dashed #2B3A5E; border-radius: 16px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 26px; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: 15px; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Modèle ───────────────────────────
def find_model():
    for name in MODEL_FILES:
        if (HERE / name).exists():
            return HERE / name
    found = sorted(HERE.glob("*.h5"))
    return found[0] if found else None


@st.cache_resource(show_spinner=False)
def load(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.resize(arr, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0


def find_conv(model):
    """Retourne (sous-modèle, couche) de la dernière convolution, que VGG16 soit imbriqué ou non."""
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            sub = find_conv(layer)
            if sub:
                return sub
        elif "Conv" in type(layer).__name__ and len(layer.output.shape) == 4:
            return model, layer
    return None


def gradcam(model, x: np.ndarray, class_idx: int):
    import tensorflow as tf
    try:
        found = find_conv(model)
        if not found:
            return None
        container, conv = found
        if container is model:
            grad_model = tf.keras.Model(model.inputs, [conv.output, model.output])
            with tf.GradientTape() as tape:
                conv_out, out = grad_model(x)
                score = out[:, class_idx]
        else:
            inner = tf.keras.Model(container.input, [conv.output, container.output])
            head = model.layers[model.layers.index(container) + 1:]
            with tf.GradientTape() as tape:
                conv_out, out = inner(x)
                tape.watch(conv_out)
                for layer in head:
                    out = layer(out)
                score = out[:, class_idx]
        grads = tape.gradient(score, conv_out)
        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
        return cam / cam.max() if cam.max() > 0 else None
    except Exception:
        return None


def overlay(img: Image.Image, cam: np.ndarray) -> Image.Image:
    base = np.array(img.convert("RGB").resize((IMG_SIZE * 2, IMG_SIZE * 2)))
    heat = cv2.resize(cam, (base.shape[1], base.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[:, :, ::-1]
    return Image.fromarray(np.uint8(0.6 * base + 0.4 * heat))


def gauge(p: float, color: str, seuil: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 44, "color": color}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
               "bar": {"color": color, "thickness": .32}, "bgcolor": "#1B2540", "borderwidth": 0,
               "threshold": {"line": {"color": TEXT, "width": 3}, "value": seuil * 100}}))
    fig.update_layout(height=230, margin=dict(t=10, b=0, l=34, r=34), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color=TEXT))
    return fig


# ─────────────────────────── Évaluation sur le dossier de test ───────────────────────────
def default_test_dir() -> str:
    for c in [HERE / "chest_xray" / "test", Path.home() / "Downloads" / "archive" / "chest_xray" / "test",
              Path.home() / "Downloads" / "chest_xray" / "test"]:
        if (c / "NORMAL").is_dir() and (c / "PNEUMONIA").is_dir():
            return str(c)
    return ""


def cache_file(model_path: Path) -> Path:
    return HERE / f"evaluation_{model_path.stem}.json"


def evaluate(model, test_dir: Path, progress) -> dict:
    files, labels = [], []
    for i, cls in enumerate(CLASSES):
        for f in sorted((test_dir / cls).iterdir()):
            if f.suffix.lower() in EXTS:
                files.append(f)
                labels.append(i)
    probs = []
    for start in range(0, len(files), 32):
        batch = np.stack([preprocess(Image.open(f)) for f in files[start:start + 32]])
        probs.extend(model.predict(batch, verbose=0)[:, 1].tolist())
        progress.progress(min(1.0, (start + 32) / len(files)), text=f"Évaluation : {min(start + 32, len(files))}/{len(files)} images")
    return {"labels": labels, "probs": probs}


def metrics(ev: dict, seuil: float) -> dict:
    y = np.array(ev["labels"])
    yp = (np.array(ev["probs"]) >= seuil).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum()); tn = int(((yp == 0) & (y == 0)).sum())
    fp = int(((yp == 1) & (y == 0)).sum()); fn = int(((yp == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {"n": len(y), "acc": div(tp + tn, len(y)), "rec": rec, "prec": prec, "spec": div(tn, tn + fp),
            "f1": div(2 * prec * rec, prec + rec), "cm": [[tn, fp], [fn, tp]]}


# ─────────────────────────── Interface ───────────────────────────
with st.sidebar:
    st.markdown("## 🫁 PneumoScan")
    st.caption("Aide à la détection de pneumonie par Deep Learning")
    st.markdown("---")
    seuil = st.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05,
                      help="Probabilité à partir de laquelle une radiographie est classée « pneumonie ». "
                           "Un seuil plus haut réduit les fausses alertes, mais peut rater des pneumonies.")
    show_cam = st.toggle("Afficher la carte Grad-CAM", value=True,
                         help="Met en évidence les zones de la radiographie qui ont le plus influencé la décision.")
    st.markdown("---")
    st.caption("Projet académique · Yousra Erraki\nEMSI · IA & Data Science")

st.markdown("""
<div class="hero">
  <div class="icon">🫁</div>
  <div>
    <div class="pill">Deep Learning · Vision par ordinateur</div>
    <h1>Pneumo<span>Scan</span></h1>
    <p>Chargez une radiographie thoracique : le modèle VGG16 estime la probabilité de pneumonie
    et montre les zones de l'image qui ont guidé sa décision.</p>
  </div>
</div>
""", unsafe_allow_html=True)

model_path = find_model()
if model_path is None:
    st.error("Aucun modèle trouvé : placez votre fichier **.h5** dans le même dossier que app.py.")
    st.stop()

with st.spinner("Chargement du modèle…"):
    model = load(str(model_path))

tab_scan, tab_model = st.tabs(["🔬  Analyse", "🧠  Le modèle"])

with tab_scan:
    fichier = st.file_uploader("Radiographie thoracique (JPG, JPEG ou PNG)", type=["jpg", "jpeg", "png"])
    if fichier is None:
        st.markdown(f'<div class="panel" style="text-align:center;padding:46px 20px;">'
                    f'<div style="font-size:46px">🩻</div>'
                    f'<h3 style="margin:8px 0 4px">Aucune radiographie chargée</h3>'
                    f'<p style="color:{MUTED};margin:0">Glissez une image ci-dessus pour lancer l\'analyse.</p></div>',
                    unsafe_allow_html=True)
    else:
        image = Image.open(io.BytesIO(fichier.getvalue()))
        with st.spinner("Analyse de la radiographie…"):
            x = np.expand_dims(preprocess(image), 0)
            probs = model.predict(x, verbose=0)[0]
            p_pneu = float(probs[1])
            malade = p_pneu >= seuil
            cam = gradcam(model, x, 1 if malade else 0) if show_cam else None

        left, right = st.columns([1.1, 1], gap="large")
        with left:
            if cam is not None:
                a, b = st.columns(2)
                a.markdown('<div class="label">Radiographie</div>', unsafe_allow_html=True)
                a.image(image.convert("RGB"), width="stretch")
                b.markdown('<div class="label">Zones d\'attention (Grad-CAM)</div>', unsafe_allow_html=True)
                b.image(overlay(image, cam), width="stretch")
            else:
                st.markdown('<div class="label">Radiographie</div>', unsafe_allow_html=True)
                st.image(image.convert("RGB"), width="stretch")

        with right:
            color = RED if malade else GREEN
            titre = "Signes de pneumonie détectés" if malade else "Aucun signe de pneumonie"
            texte = (f"Probabilité de pneumonie supérieure au seuil de {seuil:.0%}."
                     if malade else f"Probabilité de pneumonie inférieure au seuil de {seuil:.0%}.")
            st.markdown('<div class="label">Probabilité de pneumonie</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge(p_pneu, color, seuil), width="stretch")
            st.markdown(f'<div class="result" style="background:linear-gradient(120deg,{color}22,{color}0D);'
                        f'border:1px solid {color}66"><h2 style="color:{color}">{"⚠️" if malade else "✅"} {titre}</h2>'
                        f'<p>{texte}</p></div>', unsafe_allow_html=True)
            bars = "".join(
                f'<div style="display:flex;justify-content:space-between;margin-top:12px">'
                f'<span>{c.capitalize()}</span><b>{p:.1%}</b></div>'
                f'<div class="bar"><div style="width:{p*100:.1f}%;background:{GREEN if i == 0 else RED}"></div></div>'
                for i, (c, p) in enumerate(zip(CLASSES, probs)))
            st.markdown(f'<div class="panel" style="margin-top:14px"><div class="label">Détail</div>{bars}</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="disclaimer">⚕️ Projet académique : cet outil ne remplace pas l\'avis '
                        'd\'un médecin et ne doit pas être utilisé pour un diagnostic réel.</div>',
                        unsafe_allow_html=True)

with tab_model:
    cache = cache_file(model_path)
    ev = json.loads(cache.read_text()) if cache.exists() else None

    if ev is None:
        st.markdown(f'<div class="panel"><div class="label">Évaluer le modèle</div>'
                    f'<p>Les performances sont calculées sur le <b>dossier de test</b> du dataset Kaggle '
                    f'(sous-dossiers <code>NORMAL</code> et <code>PNEUMONIA</code>). Le calcul se fait une seule fois, '
                    f'puis le résultat est enregistré.</p></div>', unsafe_allow_html=True)
        test_dir = st.text_input("Chemin du dossier test", value=default_test_dir(),
                                 placeholder=r"C:\...\chest_xray\test")
        if st.button("Lancer l'évaluation", type="primary"):
            d = Path(test_dir.strip().strip('"'))
            if not ((d / "NORMAL").is_dir() and (d / "PNEUMONIA").is_dir()):
                st.error("Dossier invalide : il doit contenir les sous-dossiers NORMAL et PNEUMONIA.")
            else:
                ev = evaluate(model, d, st.progress(0.0, text="Évaluation…"))
                cache.write_text(json.dumps(ev))
                st.rerun()
    else:
        m = metrics(ev, seuil)
        st.caption(f"Modèle : {model_path.name} · {m['n']} radiographies de test · seuil de décision : {seuil:.0%}")
        k = st.columns(5)
        for col, (lab, val, sub) in zip(k, [
            ("Accuracy", m["acc"], "prédictions correctes"),
            ("Rappel pneumonie", m["rec"], "pneumonies détectées"),
            ("Précision", m["prec"], "alertes justifiées"),
            ("Spécificité", m["spec"], "poumons sains reconnus"),
            ("F1-score", m["f1"], "classe pneumonie"),
        ]):
            col.markdown(f'<div class="panel kpi"><div class="label">{lab}</div><div class="v">{val:.1%}</div>'
                         f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2, gap="large")
        cm = m["cm"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Prédit normal", "Prédit pneumonie"], y=["Réel normal", "Réel pneumonie"],
                                   text=cm, texttemplate="%{text}", textfont={"size": 20},
                                   colorscale=[[0, "#13203A"], [1, CYAN]], showscale=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Matrice de confusion", height=360, margin=dict(l=10, r=10, t=50, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", color=TEXT))
        a.plotly_chart(fig, width="stretch")
        b.markdown(f"""<div class="panel">
<div class="label">Architecture</div>
<p>• <b>VGG16</b> pré-entraîné sur ImageNet (Transfer Learning)<br>
• Tête de classification dense + Softmax (NORMAL / PNEUMONIA)<br>
• Images 224 × 224, normalisées entre 0 et 1</p>
<div class="label" style="margin-top:14px">Lecture des résultats</div>
<p>En médecine, rater une pneumonie est plus grave qu'une fausse alerte : un rappel élevé est prioritaire.
Le <b>seuil de décision</b> (barre latérale) permet de réduire les fausses alertes sur les poumons sains.</p>
<div class="label" style="margin-top:14px">Explicabilité</div>
<p><b>Grad-CAM</b> sur la dernière couche de convolution pour visualiser les zones qui influencent la décision.</p></div>""",
                   unsafe_allow_html=True)
        if st.button("Recalculer l'évaluation"):
            cache.unlink(missing_ok=True)
            st.rerun()