"""Protótipo web do classificador/localizador de faltas, via Streamlit.

Reusa a mesma função `infer_fault` usada pelo app desktop (src/infer_fault.py)
e pelos artefatos congelados em modelos/. Não retreina nada, não reimplementa
a lógica de inferência.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infer_fault import infer_fault  # noqa: E402

FREEZE_PATH = ROOT / "modelos" / "FINAL_PIPELINE_FREEZE_G2.3.json"
CLASSIFIER_PATH = ROOT / "modelos" / "robust_classifier_G2.1_calibrated.joblib"
REGRESSOR_PATH = ROOT / "modelos" / "distance_sanity_regressor.joblib"

st.set_page_config(page_title="Classificador de Faltas ATP", page_icon="⚡")
st.title("⚡ Classificador e Localizador de Faltas em Linhas de Transmissão")
st.caption("Protótipo web — envie um arquivo .pl4 gerado pelo ATP/ATPDraw.")

uploaded = st.file_uploader("Arquivo .pl4", type=["pl4"])

if uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pl4_path = Path(tmp_dir) / uploaded.name
        pl4_path.write_bytes(uploaded.getvalue())

        with st.spinner("Processando sinal e executando o modelo..."):
            try:
                result = infer_fault(
                    pl4=pl4_path,
                    classifier_path=CLASSIFIER_PATH,
                    freeze_path=FREEZE_PATH,
                    regressor_path=REGRESSOR_PATH,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falha ao processar o arquivo: {exc}")
                st.stop()

    classification = result["classification"]
    location = result["location"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classe de falta", classification["fault_class"])
        st.caption(
            f"Fração de votos: {classification['tree_vote_fraction']:.0%}"
            f" ({'conclusivo' if classification['conclusive'] else 'inconclusivo'})"
        )
    with col2:
        if location["conclusive"]:
            st.metric("Distância do PDT", f"{location['distance_from_PDT_km']:.1f} km")
            st.caption(f"Método: {location['method']}")
        else:
            st.metric("Distância do PDT", "—")
            st.caption(location["reason"])

    st.caption(f"SNR pré-falta estimado: {result['estimated_prefault_snr_db']:.1f} dB")

    with st.expander("Ver JSON completo"):
        st.json(result)
else:
    st.info("Aguardando um arquivo .pl4 para processar.")
