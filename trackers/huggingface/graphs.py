from collections import Counter
from streamlit_echarts import st_echarts  # pylint: disable=import-error
import numpy as np
import pandas as pd
import streamlit as st  # pylint: disable=import-error
import plotly.figure_factory as ff
from plotly import graph_objs as go
import plotly.express as px
from statistics import median

colors = {
    "blue": "#5470c6",
    "orange": "#FF7F0E",
    "green": "#94cc74",
    "saffron_mango": "#fac858",
    "red": "#ee6666",
    "light_blue": "#73c0de",
    "ocean_green": "#3ba272",
}
device_colors = {
    "x86": "#0071c5",
    "nvidia": "#76b900",
    "groq": "#F55036",
}


class StageCount:
    def __init__(self, df: pd.DataFrame) -> None:
        self.all_models = len(df)
        self.base_onnx = int(np.sum(df["base_onnx"]))
        self.optimized_onnx = int(np.sum(df["optimized_onnx"]))
        self.all_ops_supported = int(np.sum(df["all_ops_supported"]))
        self.fp16_onnx = int(np.sum(df["fp16_onnx"]))
        self.compiles = int(np.sum(df["compiles"]))
        self.assembles = int(np.sum(df["assembles"]))


class DeviceStageCount:
    def __init__(self, df: pd.DataFrame) -> None:
        self.all_models = len(df)
        self.base_onnx = int(np.sum(df["onnx_exported"]))
        self.optimized_onnx = int(np.sum(df["onnx_optimized"]))
        self.fp16_onnx = int(np.sum(df["onnx_converted"]))
        self.x86 = df.loc[df.x86_latency != "-", "x86_latency"].count()
        self.nvidia = df.loc[df.nvidia_latency != "-", "nvidia_latency"].count()
        self.groq = df.loc[
            df.groq_estimated_latency != "-", "groq_estimated_latency"
        ].count()


def stages_count_summary(current_df: pd.DataFrame, prev_df: pd.DataFrame) -> None:
    """
    Show count of how many models compile, assemble, etc
    """
    current = StageCount(current_df)
    prev = StageCount(prev_df)

    kpi = st.columns(7)

    kpi[0].metric(
        label="All models",
        value=current.all_models,
        delta=current.all_models - prev.all_models,
    )

    kpi[1].metric(
        label="Convert to ONNX",
        value=current.base_onnx,
        delta=current.base_onnx - prev.base_onnx,
    )

    kpi[2].metric(
        label="Optimize ONNX file",
        value=current.optimized_onnx,
        delta=current.optimized_onnx - prev.optimized_onnx,
    )

    kpi[3].metric(
        label="All ops supported",
        value=current.all_ops_supported,
        delta=current.all_ops_supported - prev.all_ops_supported,
    )

    kpi[4].metric(
        label="Convert to FP16",
        value=current.fp16_onnx,
        delta=current.fp16_onnx - prev.fp16_onnx,
    )

    kpi[5].metric(
        label="Compiles",
        value=current.compiles,
        delta=current.compiles - prev.compiles,
    )

    kpi[6].metric(
        label="Assembles",
        value=current.assembles,
        delta=current.assembles - prev.assembles,
    )

    # Show Sankey graph with percentages
    sk_val = {
        "All models": "100%",
        "Convert to ONNX": str(int(100 * current.base_onnx / current.all_models)) + "%",
        "Optimize ONNX file": str(
            int(100 * current.optimized_onnx / current.all_models)
        )
        + "%",
        "All ops supported": str(
            int(100 * current.all_ops_supported / current.all_models)
        )
        + "%",
        "Convert to FP16": str(int(100 * current.fp16_onnx / current.all_models)) + "%",
        "Compiles": str(int(100 * current.compiles / current.all_models)) + "%",
        "Assembles": str(int(100 * current.assembles / current.all_models)) + "%",
    }
    option = {
        "series": {
            "type": "sankey",
            "animationDuration": 1,
            "top": "0%",
            "bottom": "20%",
            "left": "0%",
            "right": "13.5%",
            "darkMode": "true",
            "nodeWidth": 2,
            "textStyle": {"fontSize": 16},
            "lineStyle": {"curveness": 0},
            "layoutIterations": 0,
            "layout": "none",
            "emphasis": {"focus": "adjacency"},
            "data": [
                {
                    "name": "All models",
                    "value": sk_val["All models"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Convert to ONNX",
                    "value": sk_val["Convert to ONNX"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Optimize ONNX file",
                    "value": sk_val["Optimize ONNX file"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "All ops supported",
                    "value": sk_val["All ops supported"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Convert to FP16",
                    "value": sk_val["Convert to FP16"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Compiles",
                    "value": sk_val["Compiles"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Assembles",
                    "value": sk_val["Assembles"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
            ],
            "label": {
                "position": "insideTopLeft",
                "borderWidth": 0,
                "fontSize": 16,
                "color": "white",
                "textBorderWidth": 0,
                "formatter": "{c}",
            },
            "links": [
                {
                    "source": "All models",
                    "target": "Convert to ONNX",
                    "value": current.base_onnx,
                },
                {
                    "source": "Convert to ONNX",
                    "target": "Optimize ONNX file",
                    "value": current.optimized_onnx,
                },
                {
                    "source": "Optimize ONNX file",
                    "target": "All ops supported",
                    "value": current.all_ops_supported,
                },
                {
                    "source": "All ops supported",
                    "target": "Convert to FP16",
                    "value": current.fp16_onnx,
                },
                {
                    "source": "Convert to FP16",
                    "target": "Compiles",
                    "value": current.compiles,
                },
                {
                    "source": "Compiles",
                    "target": "Assembles",
                    "value": current.assembles,
                },
            ],
        }
    }
    st_echarts(
        options=option,
        height="50px",
    )


def workload_origin(df: pd.DataFrame) -> None:
    """
    Show pie chart that groups models by author
    """
    all_authors = list(df.loc[:, "author"])
    author_count = {i: all_authors.count(i) for i in all_authors}
    all_models = len(df)

    options = {
        "darkMode": "true",
        "textStyle": {"fontSize": 16},
        "tooltip": {"trigger": "item"},
        "series": [
            {  # "Invisible" chart, used to show author labels
                "name": "Name of corpus:",
                "type": "pie",
                "radius": ["70%", "70%"],
                "data": [
                    {"value": author_count[k], "name": k} for k in author_count.keys()
                ],
                "label": {
                    "formatter": "{b}\n{d}%",
                },
            },
            {
                # Actual graph where data is shown
                "name": "Name of corpus:",
                "type": "pie",
                "radius": ["50%", "70%"],
                "data": [
                    {"value": author_count[k], "name": k} for k in author_count.keys()
                ],
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)",
                    }
                },
                "label": {
                    "position": "inner",
                    "formatter": "{c}",
                    "color": "black",
                    "textBorderWidth": 0,
                },
            },
            {
                # Show total number of models inside
                "name": "Total number of models:",
                "type": "pie",
                "radius": ["0%", "0%"],
                "data": [{"value": all_models, "name": "Total"}],
                "silent": "true",
                "label": {
                    "position": "inner",
                    "formatter": "{c}",
                    "color": "white",
                    "fontSize": 30,
                    "textBorderWidth": 0,
                },
            },
        ],
    }
    st_echarts(
        options=options,
        height="400px",
    )


def parameter_histogram(df: pd.DataFrame, show_assembled=True) -> None:
    # Add parameters histogram
    all_models = [float(x) / 1000000 for x in df["params"] if x != "-"]

    hist_data = []
    group_labels = []

    if all_models != []:
        hist_data.append(all_models)
        if show_assembled:
            group_labels.append("Models we tried compiling")
        else:
            group_labels.append("All models")

    if show_assembled:
        assembled_models = df[
            df["assembles"] == True  # pylint: disable=singleton-comparison
        ]
        assembled_models = [
            float(x) / 1000000 for x in assembled_models["params"] if x != "-"
        ]
        if assembled_models != []:
            hist_data.append(assembled_models)
            group_labels.append("Assembled models")

    if hist_data:
        fig = ff.create_distplot(
            hist_data,
            group_labels,
            bin_size=25,
            histnorm="",
            colors=list(colors.values()),
            curve_type="normal",
        )
        fig.layout.update(xaxis_title="Parameters in millions")
        fig.layout.update(yaxis_title="count")
        fig.update_xaxes(range=[1, 1000])

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown(
            """At least one model needs to reach the compiler to show this graph 😅"""
        )


def speedup_bar_chart_legacy(df: pd.DataFrame) -> None:
    """
    This function will be removed when we start getting CPU numbers for the daily tests
    """

    # Prepare data
    assembles = np.sum(df["assembles"])
    df = df[["model_name", "groq_nvidia_compute_ratio", "groq_nvidia_e2e_ratio"]]
    df = df.sort_values(by=["model_name"])
    df = df[(df.groq_nvidia_compute_ratio != "-")]
    df = df[(df.groq_nvidia_e2e_ratio != "-")]
    df["groq_nvidia_compute_ratio"] = df["groq_nvidia_compute_ratio"].astype(float)
    df["groq_nvidia_e2e_ratio"] = df["groq_nvidia_e2e_ratio"].astype(float)

    if len(df) == 0 and assembles > 0:
        st.markdown(
            (
                "We do not have GPU numbers for the model(s) mapped to the GroqChip."
                " This is potentially due to lack of out-of-the-box TensorRT support."
            )
        )
    elif assembles == 0:
        st.markdown(
            "Nothing to show here since no models have been successfully assembled."
        )
    else:
        data = [
            go.Bar(
                x=df["model_name"],
                y=df["groq_nvidia_compute_ratio"],
                name="Compute only",
            ),
            go.Bar(
                x=df["model_name"],
                y=df["groq_nvidia_e2e_ratio"],
                name="Compute + estimated I/O",
            ),
        ]

        layout = go.Layout(
            barmode="overlay",
            yaxis_title="Speedup compared to A100 GPU",
            colorway=list(colors.values()),
        )

        fig = dict(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            (
                "<sup>*</sup>Estimated I/O does NOT include delays caused by Groq's runtime. "
                "See FAQ for details."
            ),
            unsafe_allow_html=True,
        )


def speedup_text_summary_legacy(df: pd.DataFrame) -> None:
    # pylint: disable=line-too-long
    """
    This function will be removed when we start getting CPU numbers for the daily tests
    """

    # Remove empty elements and convert to float
    df = df[(df.groq_nvidia_compute_ratio != "-")]
    df = df[(df.groq_nvidia_e2e_ratio != "-")]
    df["groq_nvidia_compute_ratio"] = df["groq_nvidia_compute_ratio"].astype(float)
    df["groq_nvidia_e2e_ratio"] = df["groq_nvidia_e2e_ratio"].astype(float)

    # Show stats
    st.markdown(
        f"""<br><br><br><br><br><br>
            <p style="font-family:sans-serif; font-size: 20px;text-align: center;">Average speedup of GroqChip™ considering compute only:</p>
            <p style="font-family:sans-serif; color:{colors["blue"]}; font-size: 26px;text-align: center;"> {round(df["groq_nvidia_compute_ratio"].mean(),2)}x</p>
            <p style="font-family:sans-serif; color:{colors["blue"]}; font-size: 20px;text-align: center;"> min {round(df["groq_nvidia_compute_ratio"].min(),2)}x; median {round(median(df["groq_nvidia_compute_ratio"]),2)}x; max {round(df["groq_nvidia_compute_ratio"].max(),2)}x</p>
            <br><br>
            <p style="font-family:sans-serif; font-size: 20px;text-align: center;">Average speedup of GroqChip™ considering compute + estimated I/O<sup>*</sup>:</p>
            <p style="font-family:sans-serif; color:{colors["orange"]}; font-size: 26px;text-align: center;"> {round(df["groq_nvidia_e2e_ratio"].mean(),2)}x</p>
            <p style="font-family:sans-serif; color:{colors["orange"]}; font-size: 20px;text-align: center;"> min {round(df["groq_nvidia_e2e_ratio"].min(),2)}x; median {round(median(df["groq_nvidia_e2e_ratio"]),2)}x; max {round(df["groq_nvidia_e2e_ratio"].max(),2)}x</p>""",
        unsafe_allow_html=True,
    )


def process_latency_data(df, baseline):
    df = df[["model_name", "groq_estimated_latency", "nvidia_latency", "x86_latency"]]
    df = df.rename(columns={"groq_estimated_latency": "groq_latency"})
    df = df.sort_values(by=["model_name"])

    df.x86_latency.replace(["-"], [float("inf")], inplace=True)
    df.nvidia_latency.replace(["-"], [float("inf")], inplace=True)
    df.groq_latency.replace(["-"], [float("inf")], inplace=True)

    df["groq_latency"] = df["groq_latency"].astype(float)
    df["nvidia_latency"] = df["nvidia_latency"].astype(float)
    df["x86_latency"] = df["x86_latency"].astype(float)

    df["groq_compute_ratio"] = df[f"{baseline}_latency"] / df["groq_latency"]
    df["nvidia_compute_ratio"] = df[f"{baseline}_latency"] / df["nvidia_latency"]
    df["x86_compute_ratio"] = df[f"{baseline}_latency"] / df["x86_latency"]

    return df


def speedup_bar_chart(df: pd.DataFrame, baseline) -> None:

    if len(df) == 0:
        st.markdown(
            ("Nothing to show here since no models have been successfully benchmarked.")
        )
    else:
        df = process_latency_data(df, baseline)
        bar_chart = {}
        bar_chart["nvidia"] = go.Bar(
            x=df["model_name"],
            y=df["nvidia_compute_ratio"],
            name="NVIDIA A100",
        )
        bar_chart["groq"] = go.Bar(
            x=df["model_name"],
            y=df["groq_compute_ratio"],
            name="GroqChip 1",
        )
        bar_chart["x86"] = go.Bar(
            x=df["model_name"],
            y=df["x86_compute_ratio"],
            name="Intel(R) Xeon(R)",
        )

        # Move baseline to the back of the plot
        plot_sequence = list(bar_chart.keys())
        plot_sequence.insert(0, plot_sequence.pop(plot_sequence.index(baseline)))

        # Ensure that the baseline is the last bar
        data = [bar_chart[device_type] for device_type in plot_sequence]
        color_sequence = [device_colors[device_type] for device_type in plot_sequence]

        layout = go.Layout(
            barmode="overlay",  # group
            legend={
                "orientation": "h",
                "xanchor": "center",
                "x": 0.5,
                "y": 1.2,
            },
            yaxis_title="Latency Speedup",
            colorway=color_sequence,
            height=500,
        )

        fig = dict(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<sup>*</sup>Estimated I/O does NOT include delays caused by Groq's runtime.",
            unsafe_allow_html=True,
        )


def kpi_to_markdown(compute_ratio, device, is_baseline=False, color="#FFFFFF"):

    title = f"""<br><br>
    <p style="font-family:sans-serif; font-size: 20px;text-align: center;">Median {device} Acceleration ({len(compute_ratio)} models):</p>"""
    if is_baseline:
        return (
            title
            + f"""<p style="font-family:sans-serif; color:{color}; font-size: 26px;text-align: center;"> {1}x (Baseline)</p>"""
        )

    if len(compute_ratio) > 0:
        kpi_min, kpi_median, kpi_max = (
            round(compute_ratio.min(), 2),
            round(median(compute_ratio), 2),
            round(compute_ratio.max(), 2),
        )
    else:
        kpi_min, kpi_median, kpi_max = 0, 0, 0

    return (
        title
        + f"""<p style="font-family:sans-serif; color:{color}; font-size: 26px;text-align: center;"> {kpi_median}x</p>
    <p style="font-family:sans-serif; color:{color}; font-size: 20px;text-align: center;"> min {kpi_min}x; max {kpi_max}x</p>
    """
    )


def speedup_text_summary(df: pd.DataFrame, baseline) -> None:

    df = process_latency_data(df, baseline)

    # Some latencies are "infinite" because they could not be calculated
    # To calculate statistics, we remove all elements of df where the baseline latency is inf
    df = df[(df[baseline + "_latency"] != float("inf"))]

    # Setting latencies that could not be calculated to infinity also causes some compute ratios to be zero
    # We remove those to avoid doing any calculations with infinite latencies
    x86_compute_ratio = df["x86_compute_ratio"].to_numpy()
    nvidia_compute_ratio = df["nvidia_compute_ratio"].to_numpy()
    groq_compute_ratio = df["groq_compute_ratio"].to_numpy()
    x86_compute_ratio = x86_compute_ratio[x86_compute_ratio != 0]
    nvidia_compute_ratio = nvidia_compute_ratio[nvidia_compute_ratio != 0]
    groq_compute_ratio = groq_compute_ratio[groq_compute_ratio != 0]

    x86_text = kpi_to_markdown(
        x86_compute_ratio,
        device="Intel(R) Xeon(R) X40 CPU @ 2.00GHz",
        color=device_colors["x86"],
        is_baseline=baseline == "x86",
    )
    groq_text = kpi_to_markdown(
        groq_compute_ratio,
        device="GroqChip 1",
        color=device_colors["groq"],
        is_baseline=baseline == "groq",
    )
    nvidia_text = kpi_to_markdown(
        nvidia_compute_ratio,
        device="NVIDIA A100-PCIE-40GB",
        color=device_colors["nvidia"],
        is_baseline=baseline == "nvidia",
    )

    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"""{x86_text}""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""{nvidia_text}""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""{groq_text}""", unsafe_allow_html=True)


def compiler_errors(df: pd.DataFrame) -> None:
    compiler_errors = df[df["compiler_error"] != "-"]["compiler_error"]
    compiler_errors = Counter(compiler_errors)
    if len(compiler_errors) > 0:
        compiler_errors = pd.DataFrame.from_dict(
            compiler_errors, orient="index"
        ).reset_index()
        compiler_errors = compiler_errors.set_axis(
            ["error", "count"], axis=1, inplace=False
        )
        compiler_errors["error"] = [ce[:80] for ce in compiler_errors["error"]]
        fig = px.bar(
            compiler_errors,
            x="count",
            y="error",
            orientation="h",
            height=400,
        )
        fig.update_traces(marker_color=colors["blue"])

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""No compiler errors found :tada:""")


def io_fraction(df: pd.DataFrame) -> None:
    fig = go.Figure()
    for chips in ["1", "2", "4", "8"]:
        tmp = df[[model_entry == chips for model_entry in df["groq_chips_used"]]]
        if len(tmp) == 0:
            continue
        tmp = tmp[[model_entry != "-" for model_entry in tmp["groq_compute_latency"]]]
        if len(tmp) == 0:
            continue
        tmp = tmp[[model_entry != "-" for model_entry in tmp["groq_latency"]]]
        if len(tmp) == 0:
            continue
        print(len(tmp))
        compute_latency = tmp["groq_compute_latency"].astype("float")
        e2e_latency = tmp["groq_latency"].astype("float")

        io_fraction = 1 - compute_latency / e2e_latency
        if chips == "1":
            name = f"{chips} GroqChip ({len(tmp)} models)"
        else:
            name = f"{chips} GroqChips \n({len(tmp)} models)"
        fig.add_trace(
            go.Box(
                y=io_fraction,
                name=name,
            )
        )

    fig.layout.update(xaxis_title="Models compiled for X GroqChip Processors")
    fig.layout.update(yaxis_title="Estimated fraction of time (in %) spent on I/O")
    fig.layout.update(colorway=list(colors.values()))
    st.plotly_chart(fig, use_container_width=True)


def results_table(df: pd.DataFrame):
    model_name = st.text_input("", placeholder="Filter model by name")
    if model_name != "":
        df = df[[model_name in x for x in df["Model Name"]]]

    st.dataframe(df, height=min((len(df) + 1) * 35, 35 * 21))


def device_funnel(df: pd.DataFrame) -> None:
    """
    Show count of how many models compile, assemble, etc
    """
    summ = DeviceStageCount(df)

    stages = [
        "All models",
        "Export to ONNX",
        "Optimize ONNX file",
        "Convert to FP16",
        "Acquire Performance",
    ]
    cols = st.columns(len(stages))

    for idx, stage in enumerate(stages):
        with cols[idx]:
            st.markdown(stage)

    # Show Sankey graph with percentages
    sk_val = {
        "All models": f"{summ.all_models} models - 100%",
        "Convert to ONNX": f"{summ.base_onnx} models - "
        + str(int(100 * summ.base_onnx / summ.all_models))
        + "%",
        "Optimize ONNX file": f"{summ.optimized_onnx} models - "
        + str(int(100 * summ.optimized_onnx / summ.all_models))
        + "%",
        "Convert to FP16": f"{summ.fp16_onnx} models - "
        + str(int(100 * summ.fp16_onnx / summ.all_models))
        + "%",
        "Acquires Nvidia Perf": f"{summ.nvidia} models - "
        + str(int(100 * summ.nvidia / summ.all_models))
        + "% (Nvidia)",
        "Acquires Groq Perf": f"{summ.groq} models - "
        + str(int(100 * summ.groq / summ.all_models))
        + "% (Groq)",
        "Acquires x86 Perf": f"{summ.x86} models - "
        + str(int(100 * summ.x86 / summ.all_models))
        + "% (x86)",
    }
    option = {
        "series": {
            "type": "sankey",
            "animationDuration": 1,
            "top": "0%",
            "bottom": "20%",
            "left": "0%",
            "right": "19%",
            "darkMode": "true",
            "nodeWidth": 2,
            "textStyle": {"fontSize": 16},
            "nodeAlign": "left",
            "lineStyle": {"curveness": 0},
            "layoutIterations": 0,
            "nodeGap": 12,
            "layout": "none",
            "emphasis": {"focus": "adjacency"},
            "data": [
                {
                    "name": "All models",
                    "value": sk_val["All models"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Convert to ONNX",
                    "value": sk_val["Convert to ONNX"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Optimize ONNX file",
                    "value": sk_val["Optimize ONNX file"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Convert to FP16",
                    "value": sk_val["Convert to FP16"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Acquires Nvidia Perf",
                    "value": sk_val["Acquires Nvidia Perf"],
                    "itemStyle": {
                        "color": device_colors["nvidia"],
                        "borderColor": device_colors["nvidia"],
                    },
                },
                {
                    "name": "Acquires Groq Perf",
                    "value": sk_val["Acquires Groq Perf"],
                    "itemStyle": {
                        "color": device_colors["groq"],
                        "borderColor": device_colors["groq"],
                    },
                },
                {
                    "name": "Acquires x86 Perf",
                    "value": sk_val["Acquires x86 Perf"],
                    "itemStyle": {
                        "color": device_colors["x86"],
                        "borderColor": device_colors["x86"],
                    },
                },
            ],
            "label": {
                "position": "insideTopLeft",
                "borderWidth": 0,
                "fontSize": 16,
                "color": "white",
                "textBorderWidth": 0,
                "formatter": "{c}",
            },
            "links": [
                {
                    "source": "All models",
                    "target": "Convert to ONNX",
                    "value": summ.all_models,
                },
                {
                    "source": "Convert to ONNX",
                    "target": "Optimize ONNX file",
                    "value": summ.optimized_onnx,
                },
                {
                    "source": "Optimize ONNX file",
                    "target": "Convert to FP16",
                    "value": summ.fp16_onnx,
                },
                {
                    "source": "Convert to FP16",
                    "target": "Acquires Nvidia Perf",
                    "value": int(
                        summ.nvidia
                        * summ.fp16_onnx
                        / (summ.x86 + summ.nvidia + summ.groq)
                    ),
                },
                {
                    "source": "Convert to FP16",
                    "target": "Acquires Groq Perf",
                    "value": int(
                        summ.groq
                        * summ.fp16_onnx
                        / (summ.x86 + summ.nvidia + summ.groq)
                    ),
                },
                {
                    "source": "Convert to FP16",
                    "target": "Acquires x86 Perf",
                    "value": int(
                        summ.x86 * summ.fp16_onnx / (summ.x86 + summ.nvidia + summ.groq)
                    ),
                },
            ],
        }
    }
    st_echarts(
        options=option,
        height="70px",
    )
