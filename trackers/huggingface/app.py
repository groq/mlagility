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
            "Deep learning pioneers have been judging their progress with the Machine Learning "
            "Performance (MLPerf) inference benchmark, but have found that the corpus of models "
            "is small enough that it allows vendors to primarily compete by hand-optimizing "
            "kernels. MLAgility offers a complementary approach to MLPerf by examining the "
            "capability of vendors to provide turnkey solutions to a larger corpus of "
            "off-the-shelf models. By providing a workflow that is representative of the "
            "mass adoption customer on a variety of ML accelerators and effectively disallowing "
            "hand-crafted kernels, MLAgility bridges the gap between MLPerf and the mass adoption "
            "of hardware acceleration."
        ),
    )
    faq.add_section(
        "Why now for MLAgility?",
        (
            "Deep learning algorithms and their associated DL hardware accelerators are "
            "transitioning from early adoption into mass adoption. Production DL is now "
            "becoming available to the masses, with a desire to customize models to tackle "
            "their specific problems, and then take the path of least resistance into "
            "production. A market for turnkey solutions, starting with a model as input and "
            "provision a cost- and latency-effective acceleration solution, often in the cloud, "
            "as output, has emerged."
        ),
    )
    faq.add_section(
        "Which tool was used to generate those results?",
        (
            "All MLAgility results have been generated using the <b>benchit</b> tool v1.0.0, which is part "
            "of the MLAgility Github Repository. You can learn more about it "
            '<a href="https://github.com/groq/mlagility">here</a>.'
        ),
    )
    faq.add_section(
        "What is the experimental setup for each of the devices?",
        [
            "<b>x86</b>: Intel(R) Xeon(R) X40 CPU @ 2.00GHz on Google Cloud (custom: n2, 80 vCPU, 64.00 GiB) and OnnxRuntime version 1.14.0.",
            "<b>nvidia</b>: NVIDIA A100 40GB on Google Cloud (a2-highgpu-1g) and TensorRT version 22.12-py3.",
            "<b>groq</b>: GroqChip 1 on selfhosted GroqNode server, GroqFlow version 3.0.2 TestPyPI package, and GroqWare™ Suite version 0.9.2.",
            (
                "You can find more details about the methodology "
                '<a href="https://github.com/groq/mlagility/blob/main/docs/tools_user_guide.md">here</a>.'
            ),
        ],
    )
    faq.add_section(
        "What are the current key limitations of those results?",
        [
            (
                "Groq's latency is computed using GroqModel.estimate_latency(), which takes"
                " into account deterministic compute time and estimates an ideal runtime with"
                " ideal I/O time. It does not take into account runtime performance."
            ),
            "Results currently only represent batch 1 performance on a limited number of models, "
            "devices, vendors, and runtimes. You can learn more about future directions by reading "
            'the "What are the future directions of MLAgility?" FAQ section.',
        ],
    )
    faq.add_section(
        "What are the future directions of MLAgility?",
        [
            "Include additional classes of models (e.g. LLMs, GNNs, DLRMs).",
            "Perform experiments that include sweeps over batch and input sizes.",
            "Increase the number of devices from existing vendors (e.g. T4, A10, and H100).",
            "Include devices from additional vendors (e.g. ARM, and AMD)."
            "Include the number of runtimes supported (e.g. ORT and PyTorch for CUDA, PyTorch for x86).",
        ],
    )
    faq.add_section(
        "Who runs MLAgility?",
        (
            "MLAgility is currently maintained by the following individuals (in alphabetical order): "
            "Daniel Holanda Noronha, Jeremy Fowers, Kalin Ovtcharov, and Ramakrishnan Sivakumar."
        ),
    )
    faq.add_section(
        "License and Liability",
        (
            'THE MLAGILITY BENCHMARK IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR '
            "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, "
            "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE "
            "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER "
            "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, "
            "OUT OF OR IN CONNECTION WITH THE BENCHMARK OR THE USE OR OTHER DEALINGS IN THE "
            "BENCHMARK. Read more about it "
            '<a href="https://github.com/groq/mlagility/blob/main/LICENSE">here</a>.'
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
