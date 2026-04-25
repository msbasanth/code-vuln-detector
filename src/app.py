"""Streamlit UI for CWE Vulnerability Detector."""

import json
import os
import sys
import glob
import time

import truststore
truststore.inject_into_ssl()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.predict import load_model, predict_code, predict_file, load_qlora_model, predict_code_qlora
from src.gemma_predict import load_zero_shot_model, predict_code_zero_shot
from src.utils import load_config, get_device, load_label_map, count_parameters

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
def get_model(model_name: str, inference_only: bool = False, qlora: bool = False):
    """Load model, tokenizer, and label map once (cached per model)."""
    config = load_config("config.yaml")
    config["model_name"] = model_name  # Override with selected model
    device = get_device()
    label_map = load_label_map(config["label_map_path"])
    if qlora:
        model, tokenizer = load_qlora_model(config)
    elif inference_only:
        model, tokenizer = load_zero_shot_model(model_name)
    else:
        model = load_model(config, device)
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    return model, tokenizer, label_map, device, config


def load_model_comparison():
    """Load pre-computed model comparison metrics from JSON."""
    path = os.path.join(os.path.dirname(__file__), "..", "outputs", "model_comparison.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def measure_inference_latency(model_name: str, _model, _tokenizer, _device, max_length: int):
    """Measure average inference latency for a single model."""
    sample = 'void bad() { char *p = NULL; printf("%s", p); }'
    enc = _tokenizer(sample, max_length=max_length, padding="max_length",
                     truncation=True, return_tensors="pt")
    ids = enc["input_ids"].to(_device)
    mask = enc["attention_mask"].to(_device)
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _model(ids, mask)
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            _model(ids, mask)
            if _device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return round(np.mean(times) * 1000, 2)


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

# Load available models from config
config_base = load_config("config.yaml")
available_models = config_base.get("available_models", [
    {"name": "CodeT5-Small", "model_id": "Salesforce/codet5-small"},
    {"name": "CodeT5-Base", "model_id": "Salesforce/codet5-base"}
])
model_options = {m["name"]: m["model_id"] for m in available_models}
# Track which models are inference-only (no fine-tuning)
inference_only_ids = {m["model_id"] for m in available_models if m.get("inference_only", False)}
# Track which models are QLoRA fine-tuned
qlora_ids = {m["model_id"] for m in available_models if m.get("qlora", False)}

# --- Sidebar: navigation + branding ---
st.sidebar.title("Navigation")
nav_page = st.sidebar.radio(
    "Go to",
    ["The Detector", "Metrics", "Test Environment"],
    index=0,
    label_visibility="collapsed",
)

# --- Collapsible configuration (only shown on Detector page) ---
if nav_page == "The Detector":
    with st.expander("Configuration", expanded=False):
        cfg_col1, cfg_col2 = st.columns([3, 1])
        with cfg_col1:
            selected_model_name = st.selectbox(
                "Model",
                options=list(model_options.keys()),
                index=0,
                key="model_select",
            )
        with cfg_col2:
            top_k = st.number_input(
                "Top-K Predictions",
                min_value=1,
                max_value=20,
                value=5,
                key="top_k_input",
            )
    selected_model_id = model_options[selected_model_name]
    is_inference_only = selected_model_id in inference_only_ids
    is_qlora = selected_model_id in qlora_ids

    if is_inference_only:
        st.caption(f"Zero-shot · {selected_model_name} | 118 CWE categories")
    elif is_qlora:
        st.caption(f"QLoRA fine-tuned · {selected_model_name} | 118 CWE categories | Trained on NIST Juliet Test Suite v1.3")
    else:
        st.caption(f"Powered by {selected_model_name} | 118 CWE categories | Trained on NIST Juliet Test Suite v1.3")

    # Load model
    with st.spinner("Loading model..."):
        try:
            model, tokenizer, label_map, device, config = get_model(
                selected_model_id, inference_only=is_inference_only, qlora=is_qlora
            )
        except FileNotFoundError as e:
            st.error(f"Model error: {e}")
            if is_qlora:
                st.info(f"Please train {selected_model_name} first using: `python -m src.train_qlora`")
            else:
                st.info(f"Please train {selected_model_name} first using: `python -m src.train`")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load {selected_model_name}: {e}")
            st.stop()
else:
    # Defaults for non-detector pages (metrics/environment still need config)
    selected_model_name = list(model_options.keys())[0]
    selected_model_id = model_options[selected_model_name]
    is_inference_only = selected_model_id in inference_only_ids
    is_qlora = selected_model_id in qlora_ids
    top_k = 5
    config = load_config("config.yaml")
    label_map = load_label_map(config["label_map_path"])

# ==================== THE DETECTOR ====================
if nav_page == "The Detector":
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
                    if is_inference_only:
                        results = predict_code_zero_shot(
                            code, model, tokenizer, selected_model_id, label_map, top_k=top_k,
                        )
                    elif is_qlora:
                        results = predict_code_qlora(
                            code, model, tokenizer, label_map,
                            max_length=config["max_length"], top_k=top_k,
                        )
                    else:
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
                    if is_inference_only:
                        results = predict_code_zero_shot(
                            code, model, tokenizer, selected_model_id, label_map, top_k=top_k,
                        )
                    elif is_qlora:
                        results = predict_code_qlora(
                            code, model, tokenizer, label_map,
                            max_length=config["max_length"], top_k=top_k,
                        )
                    else:
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
                        if is_inference_only:
                            with open(fpath, "r", errors="replace") as fh:
                                file_code = fh.read()
                            results = predict_code_zero_shot(
                                file_code, model, tokenizer, selected_model_id, label_map, top_k=1,
                            )
                        elif is_qlora:
                            with open(fpath, "r", errors="replace") as fh:
                                file_code = fh.read()
                            results = predict_code_qlora(
                                file_code, model, tokenizer, label_map,
                                max_length=config["max_length"], top_k=1,
                            )
                        else:
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

# ==================== METRICS ====================
if nav_page == "Metrics":
    st.subheader("Model Comparison")
    all_comparison_data = load_model_comparison()

    trained_model_ids = {m["model_id"] for m in available_models if not m.get("inference_only", False)}

    # Training Overview: all models (training details only for fine-tuned)
    training_data = all_comparison_data if all_comparison_data else []
    # Test Performance: all models that have been evaluated
    comparison_data = all_comparison_data if all_comparison_data else []

    if training_data:
        # Training / Model Overview table
        st.markdown("**Training Overview**")
        training_rows = []
        for entry in training_data:
            is_zs = entry.get("inference_only", False)
            params = entry.get("Parameters", 0)
            size_mb = entry.get("Size (MB)", 0)
            train_time = entry.get("Training Time (s)", None)
            if isinstance(train_time, (int, float)):
                minutes, secs = divmod(int(train_time), 60)
                time_str = f"{minutes}m {secs}s"
            else:
                time_str = "N/A"
            training_rows.append({
                "Model": entry["Model"],
                "Architecture": entry.get("model_id", ""),
                "Parameters": f"{params / 1e6:.1f}M" if isinstance(params, (int, float)) and params else "N/A",
                "Model Size": f"{size_mb:.1f} MB" if isinstance(size_mb, (int, float)) and size_mb else "N/A",
                "Best Epoch": entry.get("Best Epoch", "N/A") if not is_zs else "N/A",
                "Training Time": time_str if not is_zs else "N/A",
                "Type": "Zero-shot" if is_zs else "Fine-tuned",
            })
        st.dataframe(
            pd.DataFrame(training_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Test Metrics table
        st.markdown("**Test Set Performance** _(macro-averaged)_")
        test_rows = []
        for entry in comparison_data:
            is_zs = entry.get("inference_only", False)
            model_label = entry["Model"]
            if is_zs and entry.get("eval_samples"):
                model_label += f" [{entry['eval_samples']}]"
            test_rows.append({
                "Model": model_label,
                "Accuracy": f"{entry['Accuracy']:.4f}",
                "Precision": f"{entry['Precision']:.4f}",
                "Recall": f"{entry['Recall']:.4f}",
                "F1-Score": f"{entry['F1-Score']:.4f}",
                "MCC": f"{entry['MCC']:.4f}",
                "FPR": f"{entry['FPR']:.6f}",
                "Latency (ms)": f"{entry.get('Latency (ms)', 'N/A')}",
            })
        st.dataframe(
            pd.DataFrame(test_rows),
            use_container_width=True,
            hide_index=True,
        )

        # Highlight best model (fine-tuned only for fair comparison)
        trained_entries = [e for e in comparison_data if not e.get("inference_only", False)]
        if trained_entries:
            best = max(trained_entries, key=lambda x: x["F1-Score"])
            st.success(f"Best fine-tuned model by F1-Score: **{best['Model']}** ({best['F1-Score']:.4f})")

        # Note about inference-only models
        inference_only_names = [m["name"] for m in available_models if m.get("inference_only", False)]
        if inference_only_names:
            st.info(
                f"**{', '.join(inference_only_names)}** {'is' if len(inference_only_names) == 1 else 'are'} "
                "inference-only (zero-shot, no fine-tuning) and excluded from training metrics."
            )
    else:
        st.info(
            "No model comparison data found. Run `python compare_all_metrics.py` "
            "to generate comprehensive metrics for all trained models."
        )

    # ==================== TRAINING (sub-section of Metrics) ====================
    st.divider()
    st.subheader("Dataset & Training Details")
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

# ==================== TEST ENVIRONMENT ====================
if nav_page == "Test Environment":
    st.header("Project Overview")
    st.markdown(
        "> **Source Code Vulnerability Detection Using Fine-Tuned Lightweight "
        "Large Language Models: An Efficient and Cost-Effective Approach**"
    )
    st.markdown(
        "This project focuses on source code vulnerability detection using "
        "fine-tuned lightweight large language models. The study uses the "
        "**Juliet C/C++ 1.3 Test Suite** and evaluates models such as "
        "CodeT5-base, CodeBERT-base, GraphCodeBERT-base, CodeGemma-2B, and "
        "Gemma 4 E2B IT. CodeGemma-2B is also available as a QLoRA "
        "fine-tuned model using 4-bit quantization and LoRA adapters. "
        "Performance is measured using Accuracy, Precision, "
        "Recall, F1-score, MCC, and False Positive Rate. The system is "
        "designed to support efficient and cost-effective vulnerability "
        "detection and to demonstrate inference on C/C++ code snippets."
    )

    st.divider()

    # --- Experimental Environment ---
    st.header("Experimental Environment")
    hw_col, sw_col = st.columns(2)
    with hw_col:
        st.subheader("Hardware")
        st.markdown(
            """
| Component | Specification |
|---|---|
| **Processor** | 13th Gen Intel Core i7-13850HX @ 2.10 GHz |
| **Cores / Threads** | 20 cores / 28 logical processors |
| **Integrated GPU** | Intel UHD Graphics |
| **Dedicated GPU** | NVIDIA RTX 2000 Ada Generation Laptop GPU |
| **VRAM** | ~8 GB (7957 MB) |
"""
        )
    with sw_col:
        st.subheader("Software")
        st.markdown(
            """
| Component | Details |
|---|---|
| **Language** | Python |
| **Deep Learning** | PyTorch |
| **Model Library** | Hugging Face Transformers |
| **Quantization** | bitsandbytes (NF4 4-bit) |
| **Fine-tuning** | PEFT / LoRA (QLoRA) |
| **Acceleration** | Hugging Face Accelerate |
"""
        )

    st.divider()

    # --- Training Details ---
    st.header("Training Details")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            """
| Item | Detail |
|---|---|
| **Task** | Source code vulnerability detection |
| **Dataset** | Juliet C/C++ 1.3 Test Suite |
| **Language** | C / C++ |
| **Training Objective** | Efficient and cost-effective vulnerability detection using lightweight open-source LLMs |
"""
        )
    with t2:
        st.markdown(
            """
| Item | Detail |
|---|---|
| **Candidate Models** | CodeT5-base, CodeBERT-base, GraphCodeBERT-base, CodeGemma-2B (Zero-shot & QLoRA), Gemma 4 E2B IT |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-score, MCC, False Positive Rate |
"""
        )

    st.divider()

    # --- Inference Details ---
    st.header("Inference Details")
    st.markdown(
        "Inference was performed on the same laptop environment using the "
        "**NVIDIA RTX 2000 Ada Generation Laptop GPU** with **8 GB VRAM**. "
        "For larger models such as Gemma 4 E2B IT, inference latency was "
        "affected when low-bit quantization could not be fully enabled, "
        "leading to CPU offloading. This increased execution time compared "
        "to fully GPU-resident inference."
    )
    st.info(
        "The inference module analyzes C/C++ source code snippets and predicts "
        "whether the input is vulnerable or non-vulnerable, along with the "
        "relevant vulnerability category (CWE) where applicable.",
        icon="ℹ️",
    )
