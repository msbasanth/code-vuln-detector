"""Streamlit UI for CWE Vulnerability Detector."""

import os
import sys
import glob

import truststore
truststore.inject_into_ssl()

import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.predict import load_model, predict_code, predict_file
from src.utils import load_config, get_device, load_label_map

# CWE short descriptions for the 118 classes in the Juliet Test Suite
CWE_DESCRIPTIONS = {
    "CWE-15": "External Control of System or Configuration Setting",
    "CWE-23": "Relative Path Traversal",
    "CWE-36": "Absolute Path Traversal",
    "CWE-78": "OS Command Injection",
    "CWE-90": "LDAP Injection",
    "CWE-114": "Process Control",
    "CWE-121": "Stack-based Buffer Overflow",
    "CWE-122": "Heap-based Buffer Overflow",
    "CWE-123": "Write-what-where Condition",
    "CWE-124": "Buffer Underwrite",
    "CWE-126": "Buffer Over-read",
    "CWE-127": "Buffer Under-read",
    "CWE-134": "Use of Externally-Controlled Format String",
    "CWE-176": "Improper Handling of Unicode Encoding",
    "CWE-188": "Reliance on Data/Memory Layout",
    "CWE-190": "Integer Overflow or Wraparound",
    "CWE-191": "Integer Underflow",
    "CWE-194": "Unexpected Sign Extension",
    "CWE-195": "Signed to Unsigned Conversion Error",
    "CWE-196": "Unsigned to Signed Conversion Error",
    "CWE-197": "Numeric Truncation Error",
    "CWE-222": "Truncation of Security-relevant Information",
    "CWE-223": "Omission of Security-relevant Information",
    "CWE-226": "Sensitive Information in Resource Not Removed Before Reuse",
    "CWE-242": "Use of Inherently Dangerous Function",
    "CWE-244": "Improper Clearing of Heap Data (Heap Inspection)",
    "CWE-247": "Reliance on DNS Lookups in a Security Decision",
    "CWE-252": "Unchecked Return Value",
    "CWE-253": "Incorrect Check of Function Return Value",
    "CWE-256": "Plaintext Storage of a Password",
    "CWE-259": "Use of Hard-coded Password",
    "CWE-272": "Least Privilege Violation",
    "CWE-273": "Improper Check for Dropped Privileges",
    "CWE-284": "Improper Access Control",
    "CWE-319": "Cleartext Transmission of Sensitive Information",
    "CWE-321": "Use of Hard-coded Cryptographic Key",
    "CWE-325": "Missing Cryptographic Step",
    "CWE-327": "Use of a Broken or Risky Cryptographic Algorithm",
    "CWE-328": "Use of Weak Hash",
    "CWE-338": "Use of Cryptographically Weak PRNG",
    "CWE-364": "Signal Handler Race Condition",
    "CWE-366": "Race Condition within a Thread",
    "CWE-367": "TOCTOU Race Condition",
    "CWE-369": "Divide By Zero",
    "CWE-377": "Insecure Temporary File",
    "CWE-390": "Detection of Error Condition Without Action",
    "CWE-391": "Unchecked Error Condition",
    "CWE-396": "Declaration of Catch for Generic Exception",
    "CWE-397": "Declaration of Throws for Generic Exception",
    "CWE-398": "Code Quality (7PK)",
    "CWE-400": "Uncontrolled Resource Consumption",
    "CWE-401": "Missing Release of Memory after Effective Lifetime",
    "CWE-404": "Improper Resource Shutdown or Release",
    "CWE-415": "Double Free",
    "CWE-416": "Use After Free",
    "CWE-426": "Untrusted Search Path",
    "CWE-427": "Uncontrolled Search Path Element",
    "CWE-440": "Expected Behavior Violation",
    "CWE-457": "Use of Uninitialized Variable",
    "CWE-459": "Incomplete Cleanup",
    "CWE-464": "Addition of Data Structure Sentinel",
    "CWE-467": "Use of sizeof() on a Pointer Type",
    "CWE-468": "Incorrect Pointer Scaling",
    "CWE-469": "Use of Pointer Subtraction to Determine Size",
    "CWE-475": "Undefined Behavior for Input to API",
    "CWE-476": "NULL Pointer Dereference",
    "CWE-478": "Missing Default Case in Multiple Condition Expression",
    "CWE-479": "Signal Handler Use of a Non-reentrant Function",
    "CWE-480": "Use of Incorrect Operator",
    "CWE-481": "Assigning instead of Comparing",
    "CWE-482": "Comparing instead of Assigning",
    "CWE-483": "Incorrect Block Delimitation",
    "CWE-484": "Omitted Break Statement in Switch",
    "CWE-500": "Public Static Field Not Marked Final",
    "CWE-506": "Embedded Malicious Code",
    "CWE-510": "Trapdoor",
    "CWE-511": "Logic/Time Bomb",
    "CWE-526": "Exposure of Sensitive Information Through Environmental Variables",
    "CWE-534": "Exposure of Sensitive Information Through Debug Log Files",
    "CWE-535": "Exposure of Information Through Shell Error Message",
    "CWE-546": "Suspicious Comment",
    "CWE-561": "Dead Code",
    "CWE-562": "Return of Stack Variable Address",
    "CWE-563": "Assignment to Variable without Use",
    "CWE-570": "Expression is Always False",
    "CWE-571": "Expression is Always True",
    "CWE-587": "Assignment of a Fixed Address to a Pointer",
    "CWE-588": "Attempt to Access Child of a Non-structure Pointer",
    "CWE-590": "Free of Memory not on the Heap",
    "CWE-591": "Sensitive Data Storage in Improperly Locked Memory",
    "CWE-605": "Multiple Binds to the Same Port",
    "CWE-606": "Unchecked Input for Loop Condition",
    "CWE-615": "Inclusion of Sensitive Information in Source Code Comments",
    "CWE-617": "Reachable Assertion",
    "CWE-620": "Unverified Password Change",
    "CWE-665": "Improper Initialization",
    "CWE-666": "Operation on Resource in Wrong Phase of Lifetime",
    "CWE-667": "Improper Locking",
    "CWE-672": "Operation on a Resource after Expiration or Release",
    "CWE-674": "Uncontrolled Recursion",
    "CWE-675": "Multiple Operations on Resource in Single-Operation Context",
    "CWE-676": "Use of Potentially Dangerous Function",
    "CWE-680": "Integer Overflow to Buffer Overflow",
    "CWE-681": "Incorrect Conversion between Numeric Types",
    "CWE-685": "Function Call With Incorrect Number of Arguments",
    "CWE-688": "Function Call With Incorrect Variable or Reference as Argument",
    "CWE-690": "Unchecked Return Value to NULL Pointer Dereference",
    "CWE-758": "Reliance on Undefined, Unspecified, or Implementation-Defined Behavior",
    "CWE-761": "Free of Pointer not at Start of Buffer",
    "CWE-762": "Mismatched Memory Management Routines",
    "CWE-773": "Missing Reference to Active File Descriptor or Handle",
    "CWE-775": "Missing Release of File Descriptor or Handle after Effective Lifetime",
    "CWE-780": "Use of RSA Algorithm without OAEP",
    "CWE-785": "Use of Path Manipulation Function without Maximum-sized Buffer",
    "CWE-789": "Memory Allocation with Excessive Size Value",
    "CWE-832": "Unlock of a Resource that is not Locked",
    "CWE-835": "Loop with Unreachable Exit Condition (Infinite Loop)",
    "CWE-843": "Access of Resource Using Incompatible Type (Type Confusion)",
}


@st.cache_resource
def get_model():
    """Load model, tokenizer, and label map once."""
    config = load_config("config.yaml")
    device = get_device()
    model = load_model(config, device)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    label_map = load_label_map(config["label_map_path"])
    return model, tokenizer, label_map, device, config


def get_description(cwe_id: str) -> str:
    """Get CWE description from the static dict."""
    return CWE_DESCRIPTIONS.get(cwe_id, "Unknown")


def show_results(results: list[dict]):
    """Display prediction results with chart and table."""
    if not results:
        st.warning("No predictions generated.")
        return

    top = results[0]
    st.metric("Top Prediction", top["cwe"], f"{top['confidence']:.2%} confidence")
    st.caption(get_description(top["cwe"]))

    df = pd.DataFrame(results)
    df["description"] = df["cwe"].apply(get_description)
    df["confidence_pct"] = df["confidence"].apply(lambda x: f"{x:.2%}")

    fig = px.bar(
        df,
        x="confidence",
        y="cwe",
        orientation="h",
        text="confidence_pct",
        color="confidence",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 1],
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Confidence", range=[0, 1]),
        height=max(200, len(results) * 40),
        margin=dict(l=0, r=0, t=10, b=30),
        showlegend=False,
        coloraxis_showscale=False,
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df[["cwe", "description", "confidence_pct"]].rename(
            columns={"cwe": "CWE", "description": "Description", "confidence_pct": "Confidence"}
        ),
        use_container_width=True,
        hide_index=True,
    )


@st.cache_data
def load_dataset_info():
    """Load train/test parquet metadata (cached)."""
    train_path = "data/processed/train.parquet"
    test_path = "data/processed/test.parquet"
    train_df = pd.read_parquet(train_path, columns=["file_path", "cwe_id", "cwe_name", "template_id"])
    test_df = pd.read_parquet(test_path, columns=["file_path", "cwe_id", "cwe_name", "template_id"])
    return train_df, test_df


# --- Page config ---
st.set_page_config(
    page_title="CWE Vulnerability Detector",
    page_icon="🛡️",  # noqa: RUF001
    layout="wide",
)

st.title("CWE Vulnerability Detector")
st.caption("Powered by CodeT5-small | 118 CWE categories | Trained on NIST Juliet Test Suite v1.3")

# Load model
with st.spinner("Loading model..."):
    model, tokenizer, label_map, device, config = get_model()

# Top-K slider in sidebar
top_k = st.sidebar.slider("Top-K Predictions", min_value=1, max_value=20, value=5)

# --- Main page tabs ---
page_detector, page_training = st.tabs(["The Detector", "Training"])

# ==================== THE DETECTOR ====================
with page_detector:
    tab_snippet, tab_upload, tab_folder = st.tabs(["Code Snippet", "File Upload", "Folder Scan"])

    with tab_snippet:
        code = st.text_area(
            "Paste your C/C++ code below:",
            height=300,
            placeholder="void bad() {\n    char *ptr = NULL;\n    printf(\"%s\", ptr);\n}",
        )
        if st.button("Analyze", key="analyze_snippet", type="primary"):
            if code.strip():
                with st.spinner("Analyzing..."):
                    results = predict_code(
                        code, model, tokenizer, label_map, device,
                        max_length=config["max_length"], top_k=top_k,
                    )
                show_results(results)
            else:
                st.warning("Please enter some code to analyze.")

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a C/C++ source file",
            type=["c", "cpp", "h", "hpp"],
        )
        if uploaded is not None:
            code = uploaded.getvalue().decode("utf-8", errors="replace")
            st.code(code[:2000] + ("..." if len(code) > 2000 else ""), language="c")
            if st.button("Analyze", key="analyze_upload", type="primary"):
                with st.spinner("Analyzing..."):
                    results = predict_code(
                        code, model, tokenizer, label_map, device,
                        max_length=config["max_length"], top_k=top_k,
                    )
                show_results(results)

    with tab_folder:
        folder_path = st.text_input(
            "Enter local folder path to scan:",
            placeholder=r"D:\path\to\source\files",
        )
        if st.button("Scan Folder", key="analyze_folder", type="primary"):
            if folder_path and os.path.isdir(folder_path):
                files = sorted(
                    glob.glob(os.path.join(folder_path, "**", "*.c"), recursive=True)
                    + glob.glob(os.path.join(folder_path, "**", "*.cpp"), recursive=True)
                )
                if not files:
                    st.warning("No .c/.cpp files found in the specified folder.")
                else:
                    st.info(f"Found {len(files)} file(s). Scanning...")
                    progress = st.progress(0)
                    rows = []
                    for i, fpath in enumerate(files):
                        results = predict_file(
                            fpath, model, tokenizer, label_map, device,
                            max_length=config["max_length"], top_k=1,
                        )
                        top = results[0]
                        rows.append({
                            "File": os.path.relpath(fpath, folder_path),
                            "Top CWE": top["cwe"],
                            "Confidence": f"{top['confidence']:.2%}",
                            "Description": get_description(top["cwe"]),
                        })
                        progress.progress((i + 1) / len(files))

                    progress.empty()
                    st.success(f"Scanned {len(files)} files.")
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Summary chart
                    cwe_counts = df["Top CWE"].value_counts()
                    fig = px.bar(
                        x=cwe_counts.index,
                        y=cwe_counts.values,
                        labels={"x": "CWE", "y": "File Count"},
                    )
                    fig.update_layout(margin=dict(l=0, r=0, t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter a valid folder path.")

# ==================== TRAINING ====================
with page_training:
    train_df, test_df = load_dataset_info()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train Samples", f"{len(train_df):,}")
    col2.metric("Test Samples", f"{len(test_df):,}")
    col3.metric("Total Samples", f"{len(train_df) + len(test_df):,}")
    col4.metric("CWE Classes", f"{train_df['cwe_id'].nunique()}")

    st.divider()

    # Dataset selector
    split_choice = st.radio("Select Split", ["Train", "Test"], horizontal=True, key="split_choice")
    active_df = train_df if split_choice == "Train" else test_df

    # Filters
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        all_cwes = sorted(active_df["cwe_id"].unique(), key=int)
        cwe_options = [f"CWE-{c}" for c in all_cwes]
        selected_cwe = st.selectbox("Filter by CWE", ["All"] + cwe_options, key="cwe_filter")
    with filter_col2:
        search_text = st.text_input("Search filename", placeholder="e.g. buffer_overflow", key="file_search")

    filtered = active_df.copy()
    if selected_cwe != "All":
        cwe_num = selected_cwe.replace("CWE-", "")
        filtered = filtered[filtered["cwe_id"] == cwe_num]
    if search_text:
        filtered = filtered[filtered["file_path"].str.contains(search_text, case=False, na=False)]

    # Display table
    display_df = filtered[["file_path", "cwe_id", "cwe_name", "template_id"]].copy()
    display_df.insert(1, "CWE", display_df["cwe_id"].apply(lambda x: f"CWE-{x}"))
    display_df["Description"] = display_df["CWE"].apply(get_description)
    display_df = display_df.rename(columns={
        "file_path": "File",
        "cwe_name": "CWE Name",
        "template_id": "Template ID",
    }).drop(columns=["cwe_id"])

    st.caption(f"Showing {len(filtered):,} of {len(active_df):,} samples — click a row to view file contents")
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Show selected file content
    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        row_idx = selected_rows[0]
        selected_file = display_df.iloc[row_idx]["File"]
        full_path = os.path.join(config["dataset_root"], selected_file)

        st.subheader(f"File: {selected_file}")
        if os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                file_content = f.read()
            st.code(file_content, language="c", line_numbers=True)
        else:
            st.warning(f"File not found: {full_path}")

    st.divider()

    # CWE distribution chart
    st.subheader("CWE Distribution")
    cwe_counts = active_df["cwe_id"].value_counts().reset_index()
    cwe_counts.columns = ["cwe_id", "count"]
    cwe_counts["CWE"] = cwe_counts["cwe_id"].apply(lambda x: f"CWE-{x}")
    cwe_counts = cwe_counts.sort_values("count", ascending=False)

    fig = px.bar(
        cwe_counts,
        x="CWE",
        y="count",
        color="count",
        color_continuous_scale="Blues",
        labels={"count": "Sample Count", "CWE": ""},
    )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=80),
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Template stats
    st.subheader("Template Statistics")
    template_counts = active_df.groupby("cwe_id").agg(
        samples=("file_path", "count"),
        templates=("template_id", "nunique"),
    ).reset_index()
    template_counts["CWE"] = template_counts["cwe_id"].apply(lambda x: f"CWE-{x}")
    template_counts["Description"] = template_counts["CWE"].apply(get_description)
    template_counts = template_counts[["CWE", "Description", "samples", "templates"]].rename(
        columns={"samples": "Samples", "templates": "Unique Templates"}
    ).sort_values("Samples", ascending=False)
    st.dataframe(template_counts, use_container_width=True, hide_index=True)
