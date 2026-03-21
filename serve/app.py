import streamlit as st
import pandas as pd
import base64
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from pathlib import Path
import sys

# monorepo 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mga.inference import ADMETPredictor
from mga.inference.task_registry import get_display_name
import plotly.graph_objects as go


@st.cache_resource
def get_predictor() -> ADMETPredictor:
    """모델을 한 번만 로드하고 Streamlit 세션 전반에 걸쳐 캐시."""
    return ADMETPredictor()

st.set_page_config(
    page_title="ADMET Profiling Tool",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "modal" not in st.session_state:
    st.session_state.modal = {"show": False, "task": None}


@st.dialog("Substructure Alert")
def show_substructure(task):
    st.write(f"선택된 작업: {task}")
    st.write("서브 구조 이미지가 여기에 표시됩니다.")
    if st.button("닫기"):
        st.session_state.modal["show"] = False


def decode_svg_to_html(base64_svg):
    svg_data = base64.b64decode(base64_svg).decode("utf-8")
    return f"<div style='text-align: center;'>{svg_data}</div>"


def calculate_molecular_properties(mol):
    return {
        "Molecular Weight (MW)": round(Descriptors.MolWt(mol), 2),
        "Volume": round(Descriptors.HeavyAtomMolWt(mol), 2),
        "Density": round(Descriptors.MolWt(mol) / Descriptors.HeavyAtomMolWt(mol), 3),
        "nHA": Descriptors.NumHAcceptors(mol),
        "nHD": Descriptors.NumHDonors(mol),
        "nRot": Descriptors.NumRotatableBonds(mol),
        "nRing": Descriptors.RingCount(mol),
        "TPSA": round(Descriptors.TPSA(mol), 2),
    }


def create_normalized_radar_chart(data, bounds):
    categories = list(data.keys())
    original_values = list(data.values())

    normalized_values = []
    for key, value in data.items():
        min_val, max_val = bounds[key]
        normalized_values.append((value - min_val) / (max_val - min_val) * 100)

    categories_closed = categories + [categories[0]]
    original_closed = original_values + [original_values[0]]
    normalized_closed = normalized_values + [normalized_values[0]]
    hover_texts = [f"{c}: {v}" for c, v in zip(categories_closed, original_closed)]

    fig = go.Figure(data=go.Scatterpolar(
        r=normalized_closed,
        theta=categories_closed,
        fill="toself",
        name="Normalized",
        hoverinfo="text",
        hovertext=hover_texts,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=False,
    )
    return fig


def render_section(df, category_name, subheader_title, order):
    st.subheader(subheader_title)
    category_df = df[df["category"] == category_name]
    rows = category_df[::-1].iterrows() if order == "Asc" else category_df.iterrows()
    for _, row in rows:
        with st.container():
            cols = st.columns([2, 1, 1, 0.5])
            cols[0].markdown(f"**{row['task']}**")
            cols[1].markdown(f"{row['value']}")
            cols[2].markdown(f"{row['unit']}")
            if row["threshold"] != "regression" and row["value"] >= 70:
                if cols[3].button("⚠️", key=f"alert_{category_name}_{row['task']}", help="View substructure"):
                    st.session_state.selected_task = row["task"]
                    st.session_state.show_dialog = True
                    st.rerun()


def filter_tasks_to_dict(df, task_list):
    filtered_df = df[df["task"].isin(task_list)].copy()
    if "Clearance" in filtered_df["task"].values:
        filtered_df.loc[(filtered_df["task"] == "Clearance") & (filtered_df["value"] < 0), "value"] = 0
    if "Fu" in filtered_df["task"].values:
        filtered_df.loc[(filtered_df["task"] == "Fu") & (filtered_df["value"] > 100), "value"] = 100
    return dict(zip(filtered_df["task"], filtered_df["value"]))


def run_prediction(smiles):
    result = get_predictor().predict_single(smiles, image_mode="base64")
    predictions = result[0]["Predict"]

    for pred in predictions:
        pred["task"] = get_display_name(pred["task"])

    mol = Chem.MolFromSmiles(smiles)
    df = pd.DataFrame(predictions)
    st.session_state.prediction_done = True
    st.session_state.result_df = df
    st.session_state.mol_img = Draw.MolToImage(mol, size=(400, 200))
    st.session_state.mol_properties = calculate_molecular_properties(mol)
    return df


def main():
    adme_bounds = {
        "Caco-2 Permeability": (-16.44, -3.8),
        "VDss": (-0.20, 18.57),
        "Clearance": (0, 24.23),
        "Fu": (0, 100),
    }
    property_bounds = {
        "pKa": (-16.48, 31.64),
        "pkb": (-39.17, 15.81),
        "logD": (-10.78, 10.02),
        "logP": (-13.06, 34.00),
        "logS": (-10.11, 1.66),
        "logVP": (-33.11, 60.50),
        "Hydration Energy": (-116.0, 68.00),
        "Boiling Point": (-618.41, 516.56),
        "Melting Point": (-377.89, 498.40),
    }

    if "show_dialog" not in st.session_state:
        st.session_state.show_dialog = False
    if st.session_state.modal["show"]:
        show_substructure(st.session_state.modal["task"])

    st.title("Chemiverse ADMET Prediction")
    st.markdown(
        "SMILES 문자열을 입력하면 ADMET 특성을 예측합니다."
    )

    smiles_input = st.text_input("Enter SMILES", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")

    if st.button("Run Prediction"):
        if smiles_input.strip():
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                try:
                    run_prediction(smiles_input)
                except Exception as e:
                    st.error(f"예측 실패: {str(e)}")
                    return

    if st.session_state.get("prediction_done"):
        try:
            df = st.session_state.result_df
            img = st.session_state.mol_img
            mol_properties = st.session_state.mol_properties

            adme_tasks = ["Caco-2 Permeability", "VDss", "Clearance", "Fu"]
            property_tasks = ["pKa", "pkb", "logD", "logP", "logS", "logVP", "Hydration Energy", "Boiling Point", "Melting Point"]

            adme_radar_data = filter_tasks_to_dict(df, adme_tasks)
            property_radar_data = filter_tasks_to_dict(df, property_tasks)

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.subheader("Structure")
                st.image(img, use_container_width=True)
                st.subheader("Physicochemical Properties")
                for prop, value in mol_properties.items():
                    cols = st.columns([4, 1])
                    cols[0].markdown(f"**{prop}**")
                    cols[1].markdown(f"<div style='text-align: right;'>{value}</div>", unsafe_allow_html=True)
                physchem_df = df[df["category"] == "general_properties"]
                for _, row in physchem_df.iterrows():
                    cols = st.columns([4, 1])
                    cols[0].markdown(f"**{row['task']}**")
                    cols[1].markdown(f"<div style='text-align: right;'>{row['value']}</div>", unsafe_allow_html=True)

            with col2:
                render_section(df, "absorption", "Absorption", "Desc")
                render_section(df, "distribution", "Distribution", "Desc")
                render_section(df, "metabolism", "Metabolism", "Asc")
                render_section(df, "excretion", "Excretion", "Asc")

            with col3:
                render_section(df, "toxicity", "Toxicity", "Asc")
                render_section(df, "tox21", "Tox21 Pathway", "Desc")

            st.title("Normalized Radar Charts")
            radar_col1, radar_col2 = st.columns(2)
            with radar_col1:
                st.subheader("Physicochemical Properties")
                if property_radar_data:
                    st.plotly_chart(
                        create_normalized_radar_chart(property_radar_data, property_bounds),
                        use_container_width=True,
                    )
            with radar_col2:
                st.subheader("ADME Properties")
                if adme_radar_data:
                    st.plotly_chart(
                        create_normalized_radar_chart(adme_radar_data, adme_bounds),
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"표시 오류: {e}")


if __name__ == "__main__":
    main()
