import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Shariah Compliance Portal",
    layout="wide"
)

# =========================================================
# 2. GLOBAL STYLING (LIGHT + DARK MODE FRIENDLY)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Metric Card */
.metric-card {
    padding: 22px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.2);
    background-color: rgba(255,255,255,0.03);
    backdrop-filter: blur(4px);
    text-align: center;
}

/* Status Badge */
.status-flagged {
    background-color: rgba(220, 38, 38, 0.15);
    color: #dc2626;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}

.status-passed {
    background-color: rgba(22, 163, 74, 0.15);
    color: #16a34a;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}

/* Result Banner */
.result-banner {
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    margin-top: 15px;
    margin-bottom: 20px;
    font-size: 18px;
    font-weight: 700;
    border: 1px solid rgba(128,128,128,0.2);
}

/* Table Styling */
table {
    font-size: 14px;
}

/* About Section */
.about-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.2);
    background-color: rgba(255,255,255,0.03);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. SESSION STATE
# =========================================================
if 'scan_history' not in st.session_state:
    st.session_state['scan_history'] = []

# =========================================================
# 4. LOAD MODEL & DATASET
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_dataset():

    dataset_path = r'C:\ShariahSystem\kamus.csv'

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        return df, "kamus.csv"

    else:
        st.error("Dataset file 'kamus.csv' was not found.")
        return pd.DataFrame(), "No File"

model = load_model()
df_lib, file_name = load_dataset()

# =========================================================
# 5. KNOWLEDGE BASE EMBEDDINGS
# =========================================================
kb_embeddings = None

if not df_lib.empty:
    kb_clauses = df_lib['Text_Clause'].astype(str).tolist()

    kb_embeddings = model.encode(
        kb_clauses,
        convert_to_tensor=True
    )

# =========================================================
# 6. KEYWORD DETECTION MODULE
# =========================================================
keyword_dict = {
    "Riba": [
        "interest",
        "penalty",
        "late payment",
        "per annum",
        "% charge"
    ],

    "Gharar": [
        "uncertain",
        "discretion",
        "undefined",
        "subject to"
    ],

    "Maysir": [
        "guaranteed return",
        "no risk",
        "profit guaranteed"
    ]
}

def keyword_score(sentence):

    sentence_lower = sentence.lower()

    score = 0
    detected_keywords = []
    categories = []

    for category, keywords in keyword_dict.items():

        for keyword in keywords:

            if keyword in sentence_lower:
                score += 1
                detected_keywords.append(keyword)
                categories.append(category)

    return score, detected_keywords, list(set(categories))

# =========================================================
# 7. SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("""
    <h2 style='font-weight:700;'>
    SHARIAH COMPLIANCE PORTAL
    </h2>
    """, unsafe_allow_html=True)

    st.caption("Islamic Finance Document Analysis System")

    st.divider()

    st.markdown(f"**Reference Dataset:** `{file_name}`")
    st.markdown("**Current User:** `System Administrator`")

    st.divider()

    if st.button("Clear Audit History"):
        st.session_state['scan_history'] = []
        st.rerun()

# =========================================================
# 8. MAIN TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard",
    "Reference Dataset",
    "Document Scanner",
    "Audit History",
    "About System"
])

# =========================================================
# TAB 1 — DASHBOARD
# =========================================================
with tab1:

    st.markdown("""
    <h1 style='text-align:center; font-weight:700;'>
    SHARIAH COMPLIANCE PORTAL
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h4 style='text-align:center; font-weight:400; opacity:0.7;'>
    Hybrid NLP-Based Islamic Finance Document Analysis
    </h4>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{len(st.session_state["scan_history"])}</h2>
            <p>Total Documents Analysed</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{len(df_lib)}</h2>
            <p>Knowledge Base Entries</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>Active</h2>
            <p>AI Engine Status</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2 — REFERENCE DATASET
# =========================================================
with tab2:

    st.subheader("Shariah Reference Dataset")

    st.caption("""
    Reference clauses used as the knowledge base for semantic similarity
    analysis and Shariah compliance verification.
    """)

    if not df_lib.empty:
        st.dataframe(df_lib, use_container_width=True)

    else:
        st.error("Reference dataset could not be loaded.")

# =========================================================
# TAB 3 — DOCUMENT SCANNER
# =========================================================
with tab3:

    st.subheader("Hybrid NLP Document Scanner")

    st.caption("""
    Upload Islamic finance documents for automated clause-level
    Shariah compliance analysis.
    """)

    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type="pdf"
    )

    if uploaded_file and not df_lib.empty and kb_embeddings is not None:

        if st.button("Run Hybrid NLP Analysis"):

            with st.spinner("Analysing document..."):

                # =====================================================
                # LAYER 1 — TEXT EXTRACTION
                # =====================================================
                doc = fitz.open(
                    stream=uploaded_file.read(),
                    filetype="pdf"
                )

                raw_text = " ".join([
                    page.get_text().replace('\n', ' ')
                    for page in doc
                ])

                raw_text = re.sub(r'\s+', ' ', raw_text)

                # =====================================================
                # LAYER 2 — CLAUSE FILTERING
                # =====================================================
                sentences = []

                blacklist = [
                    "tenure",
                    "amount",
                    "rm",
                    "0.00",
                    "sbr",
                    "0%"
                ]

                for sentence in re.split(r'(?<=[.!?])\s+', raw_text):

                    sentence = sentence.strip()

                    if "___" in sentence:
                        continue

                    if "date:" in sentence.lower():
                        continue

                    if (
                        len(sentence) > 60 and
                        not any(word in sentence.lower()
                                for word in blacklist)
                    ):

                        if not re.match(
                            r'^\d+(\.\d+)*\s+[A-Z\s]{5,15}$',
                            sentence
                        ):

                            sentences.append(sentence)

                # =====================================================
                # ANALYSIS PIPELINE
                # =====================================================
                if sentences:

                    kb_labels = df_lib['Label'].astype(int).tolist()
                    kb_categories = df_lib['Category'].tolist()

                    kb_reasons = df_lib[
                        'Shariah_Justification (Academic Standard)'
                    ].tolist()

                    # =================================================
                    # LAYER 3 — SEMANTIC SIMILARITY
                    # =================================================
                    input_embeddings = model.encode(
                        sentences,
                        convert_to_tensor=True
                    )

                    results = []
                    total_relevant = 0

                    for i, embedding in enumerate(input_embeddings):

                        similarity_scores = util.cos_sim(
                            embedding,
                            kb_embeddings
                        )

                        best_index = torch.argmax(
                            similarity_scores
                        ).item()

                        semantic_score = similarity_scores[
                            0
                        ][best_index].item()

                        # =============================================
                        # LAYER 4 — KEYWORD DETECTION
                        # =============================================
                        kw_score, kw_found, kw_category = keyword_score(
                            sentences[i]
                        )

                        # =============================================
                        # LAYER 5 — FUSION LOGIC
                        # =============================================
                        final_score = (
                            (0.7 * semantic_score) +
                            (0.3 * (kw_score > 0))
                        )

                        # =============================================
                        # THRESHOLD FILTER
                        # =============================================
                        if final_score >= 0.48:

                            total_relevant += 1

                            label = kb_labels[best_index]
                            category = kb_categories[best_index]

                            # =========================================
                            # RULE-BASED OVERRIDE
                            # =========================================
                            if (
                                kw_score > 0 and
                                any(
                                    word in sentences[i].lower()
                                    for word in [
                                        "interest",
                                        "compound"
                                    ]
                                )
                            ):
                                label = 0
                                category = "Riba"

                            status = (
                                "PASSED"
                                if label == 1
                                else "FLAGGED"
                            )

                            badge_class = (
                                "status-passed"
                                if status == "PASSED"
                                else "status-flagged"
                            )

                            # =========================================
                            # STORE ONLY FLAGGED CLAUSES
                            # =========================================
                            if status == "FLAGGED":

                                results.append({

                                    "Document Clause":
                                        sentences[i][:150] + "...",

                                    "Risk Category":
                                        category,

                                    "Compliance Status":
                                        f'<span class="{badge_class}">'
                                        f'{status}</span>',

                                    "Fusion Score":
                                        f"{final_score:.1%}",

                                    "Detected Keywords":
                                        ", ".join(kw_found)
                                        if kw_found else "None",

                                    "Shariah Justification":
                                        kb_reasons[best_index]
                                })

                    # =================================================
                    # OUTPUT RESULTS
                    # =================================================
                    flagged_count = len(results)

                    verdict = (
                        "FLAGGED"
                        if flagged_count > 0
                        else "PASSED"
                    )

                    # Save history
                    st.session_state['scan_history'].append({

                        "Scan Time":
                            datetime.now().strftime("%H:%M:%S"),

                        "Document":
                            uploaded_file.name,

                        "Result":
                            verdict,

                        "Flagged Clauses":
                            flagged_count
                    })

                    # =================================================
                    # DISPLAY RESULTS
                    # =================================================
                    if flagged_count > 0:

                        st.markdown(f"""
                        <div class="result-banner"
                        style="
                        background-color: rgba(220,38,38,0.12);
                        color:#dc2626;
                        ">
                        SHARIAH RISK DETECTED
                        ({flagged_count} FLAGGED CLAUSES)
                        </div>
                        """, unsafe_allow_html=True)

                        st.write(
                            pd.DataFrame(results).to_html(
                                escape=False,
                                index=False
                            ),
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown("""
                        <div class="result-banner"
                        style="
                        background-color: rgba(22,163,74,0.12);
                        color:#16a34a;
                        ">
                        DOCUMENT PASSED SHARIAH COMPLIANCE ANALYSIS
                        </div>
                        """, unsafe_allow_html=True)

                        st.success("""
                        No suspicious Shariah-related clauses
                        were detected in the uploaded document.
                        """)

                    # =================================================
                    # EXECUTIVE SUMMARY
                    # =================================================
                    st.info(f"""
                    Executive Summary

                    • Total Relevant Clauses Analysed:
                    {total_relevant}

                    • Total Flagged Clauses:
                    {flagged_count}

                    • Estimated Risk Level:
                    {(flagged_count / max(1, total_relevant)) * 100:.1f}%
                    """)

                else:

                    st.warning("""
                    The system could not extract sufficient readable
                    text from the uploaded document.
                    """)

# =========================================================
# TAB 4 — AUDIT HISTORY
# =========================================================
with tab4:

    st.subheader("Document Audit History")

    st.caption("""
    Temporary audit trail generated during the current session.
    """)

    if st.session_state['scan_history']:

        st.table(
            pd.DataFrame(
                st.session_state['scan_history']
            )
        )

    else:
        st.info("No audit history available.")

# =========================================================
# TAB 5 — ABOUT SYSTEM
# =========================================================
with tab5:

    st.subheader("About System")

    st.markdown("""
    <div class="info-card">
        <h3>System Overview</h3>
        <p>
        The Shariah Compliance Portal is a hybrid NLP-based prototype
        developed to support automated Shariah compliance verification
        for Islamic home financing documents. The system combines
        semantic similarity analysis and keyword-based detection to
        identify potentially non-compliant clauses related to
        Riba, Gharar, and Maysir.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Project Objectives</h3>
        <p>- Develop a prototype system for automated Shariah compliance verification.</p>
        <p>- Analyse Islamic home financing clauses using semantic similarity and keyword detection.</p>
        <p>- Improve transparency and consistency in Shariah auditing processes.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>System Features</h3>
        <p>- PDF document upload and text extraction.</p>
        <p>- Clause-level semantic similarity analysis.</p>
        <p>- Keyword-based Shariah risk detection.</p>
        <p>- Fusion scoring for clause classification.</p>
        <p>- Flagged clause highlighting with justification notes.</p>
        <p>- Temporary audit history tracking.</p>
    </div>
    """, unsafe_allow_html=True)
