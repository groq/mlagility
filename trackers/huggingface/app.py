from os import listdir
from os.path import isfile, join
import pandas as pd
import streamlit as st  # pylint: disable=import-error
import graphs
from streamlit_helpers import add_filter, slider_filter, Collapsable

st.set_page_config(
    page_title="ML Agility tracker",
    page_icon="⚡",
    layout="wide",
)

# dashboard title
st.title("ML Agility tracker ⚡")


def add_faq() -> None:
    """
    Displays FAQ using Collapsable sections
    """
    faq = Collapsable()
    faq.add_section(
        "How is MLAgility different from MLPerf?",
        (
            "Deep learning pioneers have been judging their progress with the Machine "
            "Learning Performance (MLPerf) inference benchmark. However, the corpus of "
            "models in MLPerf is small enough that vendors primarily compete by paying "
            "computer architecture experts to exhaustively hand-optimize kernels. This "
            "workflow is not representative of the turnkey workflow of the mass adoption "
            "customer, as they are neither likely to deploy off-the-shelf research models "
            "as-is, nor manually optimize their models by hand-writing kernels for "
            "hardware accelerators. MLAgility addresses the disconnect between MLPerf and "
            "mass adoption customers by examining a larger corpus of models, while "
            "disallowing hand-written kernels."
        ),
    )
    faq.add_section(
        "Who runs MLAgility?",
        (
            "MLAgility is lead and maintained by a consortium of industry leaders and innovators."
        ),
    )
    faq.add_section(
        "Why now for MLAgility?",
        (
            "Deep learning (DL) algorithms and their associated DL hardware accelerators are "
            "transitioning from early adoption into mass adoption. Production DL is now "
            "becoming available to the masses, with a desire to customize models to tackle "
            "their specific problems, and then take the path of least resistance into "
            "production. A market for turnkey solutions, starting with a model as input and "
            "provision a cost- and latency-effective acceleration solution, often in the cloud, "
            "as output, has emerged."
        ),
    )
    faq.add_section(
        "What are the current key limitations of MLAgility?",
        (
            "Groq's latency is computed using GroqModel.estimate_latency(), which takes"
            " into account deterministic compute time and estimates an ideal runtime with"
            " ideal I/O time. It does not take into account runtime performance."
        ),
    )

    faq.deploy()


# Add all filters to sidebar
with st.sidebar:

    st.markdown("# Filters")

    # Get all reports of a given test type
    REPORT_FOLDER = "reports"
    reports = sorted(
        [f for f in listdir(REPORT_FOLDER) if isfile(join(REPORT_FOLDER, f))]
    )

    # Select and read a report
    selected_report = st.selectbox("Test date", reports, index=len(reports) - 1)
    selected_report_idx = reports.index(selected_report)
    report = pd.read_csv(f"{REPORT_FOLDER}/{selected_report}")

    # Convert int parameters to int/float
    for p in ["groq_chips_used", "params"]:
        report[p] = report[p].replace("-", 0).astype("int64")

    # Add parameter filter
    st.markdown("#### Parameters")

    report = slider_filter(
        [report], "Select a range parameters (in millions)", filter_by="params"
    )[0]

    # Add author filter
    report = add_filter(
        [report],
        "Origin",
        label="author",
        num_cols=2,
    )[0]

    # Add task filter
    report = add_filter([report], "Tasks", label="task", options=None)[0]


st.markdown("## Summary Results")

cols = st.columns(2)
with cols[0]:
    st.markdown("""#### Workload origin""")
    graphs.workload_origin(report)

with cols[1]:
    st.markdown("""#### Parameter Size Distribution""")
    graphs.parameter_histogram(report, show_assembled=False)


st.markdown("""#### Benchmark results""")
baseline = st.selectbox("Baseline", ("x86", "nvidia", "groq"))
graphs.speedup_text_summary(report, baseline)
graphs.speedup_bar_chart(report, baseline)

# FAQ Block
cols = st.columns(2)
with cols[0]:

    st.markdown("""## About this workload analysis (FAQ)""")
    add_faq()

# Detailed data view (table)
st.markdown("## Detailed Data View")

# Add columns that do not exist yet
report["gpu_chips_used"] = 1
report["cpu_chips_used"] = 1


# Using 3 significant digits
report["groq_estimated_latency"] = [
    "-" if x == "-" else "{:.3f}".format(float(x))
    for x in report["groq_estimated_latency"]
]
report["nvidia_latency"] = [
    "-" if x == "-" else "{:.3f}".format(float(x)) for x in report["nvidia_latency"]
]
report["x86_latency"] = [
    "-" if x == "-" else "{:.3f}".format(float(x)) for x in report["x86_latency"]
]

renamed_cols = {
    "model_name": "Model Name",
    "author": "Source",
    "params": "Parameters",
    "groq_estimated_latency": "GroqChip 1: Latency (ms)",
    "nvidia_latency": "NVIDIA A100-PCIE-40GB: Latency (ms)",
    "x86_latency": "Intel(R) Xeon(R) x40 CPU: Latency (ms)",
    "groq_chips_used": "GroqChip 1: Chips Used",
    "gpu_chips_used": "NVIDIA A100-PCIE-40GB: Chips Used",
    "cpu_chips_used": "Intel(R) Xeon(R) x40 CPU: Chips Used",
}

report.rename(columns=renamed_cols, inplace=True)
selected_cols = list(renamed_cols.values())

graphs.results_table(report[selected_cols])  # pylint: disable=unsubscriptable-object
